#include "learning/ntuple_network.hpp"

#include "learning/relaxed_atomic.hpp"

#include <algorithm>
#include <array>
#include <cstring>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace adversarial_2048::learning {
namespace {

constexpr std::size_t kMaxTupleCells = 8;  // 16^8 already exceeds any sane budget
constexpr char kMagic[8] = {'A', '2', '0', '4', '8', 'N', 'T', 'N'};
// Version 1 files predate multi-stage networks and are implicitly 1 stage;
// they must keep loading, so the reader accepts both versions and only writes
// version 2 when there is genuinely more than one stage.
constexpr std::uint32_t kFormatVersion = 1;
constexpr std::uint32_t kStagedFormatVersion = 2;
// Version 3 adds a flags word. Versions 1 and 2 stay readable, so every weight
// file produced before this change remains valid.
constexpr std::uint32_t kFlaggedFormatVersion = 3;
constexpr std::uint32_t kFlagGlobalFeatures = 1U << 0U;
constexpr std::uint32_t kFlagRelativeIndexing = 1U << 1U;
// Upper byte carries the stage split point. Zero means "not recorded", which
// maps to the historical default of 10, so v3 files written before this change
// still load with their original meaning.
constexpr std::uint32_t kFlagStructuralFeatures = 1U << 2U;
constexpr std::uint32_t kFlagRelativeBank = 1U << 3U;
// Bits 8-15 carry the stage split point, so flag bits stop at 7.
constexpr std::uint32_t kStageBaseShift = 8U;

// The 8 dihedral symmetries of the square, as (row, column) maps. A tuple's
// symmetric ordering is produced by applying one of these to each of its base
// cells while PRESERVING ORDER, so all 8 orderings index the same LUT.
using CellMapper = std::pair<std::size_t, std::size_t> (*)(std::size_t, std::size_t);

constexpr std::size_t kLast = kBoardWidth - 1;

std::pair<std::size_t, std::size_t> map_identity(std::size_t r, std::size_t c) { return {r, c}; }
std::pair<std::size_t, std::size_t> map_rot90(std::size_t r, std::size_t c) { return {c, kLast - r}; }
std::pair<std::size_t, std::size_t> map_rot180(std::size_t r, std::size_t c) { return {kLast - r, kLast - c}; }
std::pair<std::size_t, std::size_t> map_rot270(std::size_t r, std::size_t c) { return {kLast - c, r}; }
std::pair<std::size_t, std::size_t> map_flip_lr(std::size_t r, std::size_t c) { return {r, kLast - c}; }
std::pair<std::size_t, std::size_t> map_flip_ud(std::size_t r, std::size_t c) { return {kLast - r, c}; }
std::pair<std::size_t, std::size_t> map_transpose(std::size_t r, std::size_t c) { return {c, r}; }
std::pair<std::size_t, std::size_t> map_anti_transpose(std::size_t r, std::size_t c) {
    return {kLast - c, kLast - r};
}

constexpr std::array<CellMapper, 8> kSymmetries{
    map_identity, map_rot90, map_rot180, map_rot270,
    map_flip_lr,  map_flip_ud, map_transpose, map_anti_transpose};

// Tuple indices pack 4 bits per cell, but Board supports exponents up to 31
// via `exponent_high_bits`. Clamp anything at or above 16 down to 15, so a
// 65536 tile indexes identically to a 32768 tile. The reference
// implementations cap at 32768 and never face this case; we must handle it
// explicitly or a high-tile board would index the wrong weights entirely.
[[nodiscard]] std::uint64_t clamped_packed(Board board) noexcept {
    if (board.exponent_high_bits == 0) {
        return board.packed_exponents;
    }
    auto packed = board.packed_exponents;
    for (std::size_t cell = 0; cell < kCellCount; ++cell) {
        if (((board.exponent_high_bits >> cell) & 1U) != 0U) {
            packed |= std::uint64_t{0xF} << (4U * cell);  // force the nibble to 15
        }
    }
    return packed;
}

[[nodiscard]] std::size_t lut_size_for(std::size_t cell_count) {
    std::size_t size = 1;
    for (std::size_t index = 0; index < cell_count; ++index) {
        size *= 16U;
    }
    return size;
}

// Rank-relative cell codes for the relative bank, packed 4 bits per cell.
//
//   empty       -> 0
//   occupied    -> 1 + min(7, board_max_exponent - exponent)
//
// Nine states, which is why the bank indexes base 9 rather than base 16. The
// drop is clamped at 7 because cells more than seven doublings below the
// maximum carry no information about the large-tile structure this bank exists
// to generalise over, and separate codes for them would only fragment the
// table.
[[nodiscard]] std::uint64_t relative_bank_codes(std::uint64_t clamped) noexcept {
    std::uint64_t maximum = 0;
    for (std::size_t cell = 0; cell < kCellCount; ++cell) {
        maximum = std::max(maximum, (clamped >> (4U * cell)) & 0xFULL);
    }
    std::uint64_t codes = 0;
    for (std::size_t cell = 0; cell < kCellCount; ++cell) {
        const auto exponent = (clamped >> (4U * cell)) & 0xFULL;
        if (exponent == 0) {
            continue;  // code 0, already there
        }
        const auto drop = std::min<std::uint64_t>(7U, maximum - exponent);
        codes |= (drop + 1U) << (4U * cell);
    }
    return codes;
}

void write_exact(std::ofstream& stream, const void* data, std::size_t bytes) {
    if (bytes > 0) {
        stream.write(static_cast<const char*>(data), static_cast<std::streamsize>(bytes));
    }
}

void read_exact(std::ifstream& stream, void* data, std::size_t bytes,
                const std::filesystem::path& path) {
    if (bytes == 0) {
        return;
    }
    stream.read(static_cast<char*>(data), static_cast<std::streamsize>(bytes));
    if (!stream) {
        throw std::runtime_error("truncated n-tuple weight file: " + path.string());
    }
}

}  // namespace

std::vector<TupleSpec> default_tuple_specs() {
    // The 2014 "large" network (Szubert & Jaskowski): two straight 4-tuples
    // and two 2x3 rectangles. Cells are row-major indices.
    return {
        TupleSpec{{0, 1, 2, 3}},                 // top row
        TupleSpec{{4, 5, 6, 7}},                 // second row
        TupleSpec{{0, 1, 2, 4, 5, 6}},           // 2x3 rectangle, rows 0-1
        TupleSpec{{4, 5, 6, 8, 9, 10}},          // 2x3 rectangle, rows 1-2
    };
}

std::vector<TupleSpec> named_tuple_specs(const std::string& name) {
    if (name == "default") {
        return default_tuple_specs();
    }
    if (name == "large") {
        // Five distinct 6-tuples: 5 * 16^6 = 83,886,080 weights (~320 MB).
        //
        // The published networks at this size are described by shape rather
        // than by explicit cell lists, so these particular sets are OUR choice,
        // not a transcription. The selection principle is coverage variety —
        // two "axe" shapes that straddle a row boundary, two horizontal 2x3
        // rectangles, and one vertical 3x2 — so that different tuples fail on
        // different board configurations rather than all sharing a blind spot.
        //
        // Only the left/top region is covered explicitly; dihedral weight
        // sharing supplies the other orientations for free.
        return {
            TupleSpec{{0, 1, 2, 3, 4, 5}},      // top row plus two of the next
            TupleSpec{{4, 5, 6, 7, 8, 9}},      // second row plus two of the third
            TupleSpec{{0, 1, 2, 4, 5, 6}},      // 2x3 rectangle, rows 0-1
            TupleSpec{{4, 5, 6, 8, 9, 10}},     // 2x3 rectangle, rows 1-2
            TupleSpec{{0, 1, 4, 5, 8, 9}},      // 3x2 rectangle, columns 0-1
        };
    }
    if (name == "xlarge") {
        // Eight distinct 6-tuples: 8 * 16^6 = 134,217,728 weights (~512 MB).
        // Systematic coverage rather than ad hoc shapes: all four 2x3
        // rectangle positions in the top-left quadrant, two 3x2 rectangles,
        // and two row-plus-overhang "axe" shapes. Dihedral weight sharing
        // supplies the remaining orientations, so covering one quadrant covers
        // the board.
        return {
            TupleSpec{{0, 1, 2, 3, 4, 5}},      // row 0 + two of row 1
            TupleSpec{{4, 5, 6, 7, 8, 9}},      // row 1 + two of row 2
            TupleSpec{{0, 1, 2, 4, 5, 6}},      // 2x3, rows 0-1, cols 0-2
            TupleSpec{{1, 2, 3, 5, 6, 7}},      // 2x3, rows 0-1, cols 1-3
            TupleSpec{{4, 5, 6, 8, 9, 10}},     // 2x3, rows 1-2, cols 0-2
            TupleSpec{{5, 6, 7, 9, 10, 11}},    // 2x3, rows 1-2, cols 1-3
            TupleSpec{{0, 1, 4, 5, 8, 9}},      // 3x2, rows 0-2, cols 0-1
            TupleSpec{{1, 2, 5, 6, 9, 10}},     // 3x2, rows 0-2, cols 1-2
        };
    }
    std::string valid;
    for (const auto& option : tuple_configuration_names()) {
        if (!valid.empty()) {
            valid += ", ";
        }
        valid += option;
    }
    throw std::invalid_argument("unknown tuple configuration '" + name + "' (valid: " +
                                valid + ")");
}

std::vector<std::string> tuple_configuration_names() {
    return {"default", "large", "xlarge"};
}

std::size_t relative_bank_lut_size(std::size_t cell_count) noexcept {
    std::size_t size = 1;
    for (std::size_t index = 0; index < cell_count; ++index) {
        size *= 9U;
    }
    return size;
}

std::size_t stage_of(Board board, std::size_t stage_count,
                     std::uint8_t base_exponent) noexcept {
    if (stage_count <= 1) {
        return 0;
    }
    // Max exponent never decreases during a game, so a game moves forward
    // through stages and never back.
    const auto exponent = max_exponent(board);
    if (exponent < base_exponent) {
        return 0;
    }
    const auto stage = static_cast<std::size_t>(exponent - base_exponent) + 1U;
    return stage < stage_count ? stage : stage_count - 1;
}

std::size_t global_feature_index(Board board) noexcept {
    const auto empties = empty_count(board);
    auto exponent = static_cast<std::size_t>(max_exponent(board));
    if (exponent > 15) {
        exponent = 15;  // same clamping rule as the tuple indices
    }
    return empties * 16U + exponent;
}

// Shifts every exponent so the board maximum becomes 15, clamping at 0. A board
// holding {16384, 8192, 4096} and one holding {2048, 1024, 512} then produce
// identical indices, so experience transfers across scales.
[[nodiscard]] std::uint64_t relative_packed(std::uint64_t clamped) noexcept {
    std::uint64_t maximum = 0;
    for (std::size_t cell = 0; cell < kCellCount; ++cell) {
        maximum = std::max(maximum, (clamped >> (4U * cell)) & 0xFULL);
    }
    if (maximum == 0) {
        return clamped;  // empty board: nothing to normalise against
    }
    const auto shift = 15ULL - maximum;
    std::uint64_t result = 0;
    for (std::size_t cell = 0; cell < kCellCount; ++cell) {
        const auto exponent = (clamped >> (4U * cell)) & 0xFULL;
        // Empty stays empty; occupied cells shift up toward 15 and saturate at
        // 1 so a tile never becomes indistinguishable from an empty cell.
        const auto mapped = exponent == 0 ? 0ULL : std::min(15ULL, exponent + shift);
        result |= mapped << (4U * cell);
    }
    return result;
}

std::size_t structural_feature_index(Board board) noexcept {
    const auto cells = decode(board);

    // How far the descending order holds along the boustrophedon path. This is
    // the "snake" every strong 2048 policy maintains, and no tuple can see it
    // because it spans the whole board.
    static constexpr std::array<std::size_t, 16> kSnake{
        0, 1, 2, 3, 7, 6, 5, 4, 8, 9, 10, 11, 15, 14, 13, 12};
    std::size_t run = 0;
    std::uint8_t previous = 16;  // above any real exponent
    for (const auto cell : kSnake) {
        const auto exponent = cells[cell];
        if (exponent == 0 || exponent > previous) {
            break;
        }
        previous = exponent;
        ++run;
    }
    if (run > 15) {
        run = 15;
    }

    // Is the largest tile in a corner? Losing the corner is how the snake
    // collapses.
    const auto peak = max_exponent(board);
    const bool cornered = peak != 0 && (cells[0] == peak || cells[3] == peak ||
                                        cells[12] == peak || cells[15] == peak);

    auto empties = empty_count(board);
    if (empties > 15) {
        empties = 15;
    }
    return run * 32U + (cornered ? 16U : 0U) + empties;
}

NTupleNetwork::NTupleNetwork(std::vector<TupleSpec> specs, std::size_t stage_count,
                             bool global_features, IndexingMode indexing,
                             std::uint8_t stage_base_exponent, bool structural_features,
                             bool relative_bank)
    : specs_(std::move(specs)), stage_count_(stage_count), global_features_(global_features),
      indexing_(indexing), stage_base_exponent_(stage_base_exponent),
      structural_features_(structural_features), relative_bank_(relative_bank) {
    if (specs_.empty()) {
        throw std::invalid_argument("n-tuple network needs at least one tuple");
    }
    if (stage_count_ == 0) {
        throw std::invalid_argument("n-tuple network needs at least one stage");
    }

    std::size_t offset = 0;
    for (const auto& spec : specs_) {
        if (spec.cells.empty()) {
            throw std::invalid_argument("n-tuple cannot be empty");
        }
        if (spec.cells.size() > kMaxTupleCells) {
            throw std::invalid_argument("n-tuple has too many cells (max " +
                                        std::to_string(kMaxTupleCells) + ")");
        }
        for (const auto cell : spec.cells) {
            if (cell >= kCellCount) {
                throw std::invalid_argument("n-tuple cell index out of range: " +
                                            std::to_string(cell));
            }
        }

        Tuple tuple;
        tuple.lut_offset = offset;
        tuple.lut_size = lut_size_for(spec.cells.size());

        // Build each symmetric ordering, dropping duplicates. Duplicates
        // genuinely occur for symmetric shapes (e.g. a 2x2 square maps onto
        // itself under some elements), and the reference drops them too.
        std::vector<std::vector<std::uint8_t>> orderings;
        for (const auto mapper : kSymmetries) {
            std::vector<std::uint8_t> mapped;
            mapped.reserve(spec.cells.size());
            for (const auto cell : spec.cells) {
                const auto row = static_cast<std::size_t>(cell) / kBoardWidth;
                const auto column = static_cast<std::size_t>(cell) % kBoardWidth;
                const auto [mapped_row, mapped_column] = mapper(row, column);
                mapped.push_back(
                    static_cast<std::uint8_t>(mapped_row * kBoardWidth + mapped_column));
            }
            if (std::find(orderings.begin(), orderings.end(), mapped) == orderings.end()) {
                orderings.push_back(std::move(mapped));
            }
        }

        // Store nibble shifts rather than cell indices so the hot path skips
        // a multiply per lookup.
        for (const auto& ordering : orderings) {
            std::vector<std::uint8_t> shifts;
            shifts.reserve(ordering.size());
            for (const auto cell : ordering) {
                shifts.push_back(static_cast<std::uint8_t>(4U * cell));
            }
            tuple.ordering_shifts.push_back(std::move(shifts));
        }

        active_weight_count_ += tuple.ordering_shifts.size();
        offset += tuple.lut_size;
        tuples_.push_back(std::move(tuple));
    }

    // The global table lives inside each stage, so a staged network gets one
    // per stage -- consistent with every other weight here.
    global_offset_ = offset;
    if (global_features_) {
        offset += kGlobalFeatureSize;
        ++active_weight_count_;
    }
    structural_offset_ = offset;
    if (structural_features_) {
        offset += kStructuralFeatureSize;
        ++active_weight_count_;
    }
    // The relative bank goes LAST so that global_offset_ -- and therefore
    // tuple_weight_count() and adopt_tuple_weights() -- keep meaning "the
    // absolute tuple region". That is what lets a trained absolute network be
    // resumed into a bank-enabled one with the bank starting at zero.
    if (relative_bank_) {
        for (std::size_t index = 0; index < tuples_.size(); ++index) {
            tuples_[index].relative_offset = offset;
            tuples_[index].relative_size = relative_bank_lut_size(specs_[index].cells.size());
            offset += tuples_[index].relative_size;
            active_weight_count_ += tuples_[index].ordering_shifts.size();
        }
    }
    stage_stride_ = offset;
    weights_.assign(offset * stage_count_, 0.0F);
}

std::uint64_t NTupleNetwork::indexed_packed(Board board) const noexcept {
    const auto clamped = clamped_packed(board);
    return indexing_ == IndexingMode::relative ? relative_packed(clamped) : clamped;
}

std::size_t NTupleNetwork::stage_offset(Board board) const noexcept {
    return stage_count_ == 1
        ? 0
        : stage_of(board, stage_count_, stage_base_exponent_) * stage_stride_;
}

std::size_t NTupleNetwork::tuple_weight_count() const noexcept {
    return global_offset_;  // feature tables begin where the tuple LUTs end
}

void NTupleNetwork::adopt_tuple_weights(const NTupleNetwork& source) {
    if (source.specs_.size() != specs_.size()) {
        throw std::invalid_argument("adopt_tuple_weights: different tuple count");
    }
    for (std::size_t index = 0; index < specs_.size(); ++index) {
        if (source.specs_[index].cells != specs_[index].cells) {
            throw std::invalid_argument("adopt_tuple_weights: different tuple cells");
        }
    }
    if (source.stage_count_ != stage_count_) {
        throw std::invalid_argument("adopt_tuple_weights: different stage count");
    }
    if (source.indexing_ != indexing_) {
        throw std::invalid_argument("adopt_tuple_weights: different indexing mode");
    }
    const auto span = std::min(source.tuple_weight_count(), tuple_weight_count());
    for (std::size_t stage = 0; stage < stage_count_; ++stage) {
        const auto* from = source.weights_.data() + stage * source.stage_stride_;
        auto* to = weights_.data() + stage * stage_stride_;
        std::copy(from, from + span, to);
    }
}

void NTupleNetwork::replicate_stage_zero() {
    for (std::size_t stage = 1; stage < stage_count_; ++stage) {
        std::copy(weights_.begin(), weights_.begin() + static_cast<std::ptrdiff_t>(stage_stride_),
                  weights_.begin() + static_cast<std::ptrdiff_t>(stage * stage_stride_));
    }
}

std::size_t NTupleNetwork::index_of(
    std::uint64_t packed, const std::vector<std::uint8_t>& shifts) const noexcept {
    // Most-significant-first packing, matching the reference's
    // `address = address * 16 + value`.
    std::size_t index = 0;
    for (const auto shift : shifts) {
        index = (index << 4U) | static_cast<std::size_t>((packed >> shift) & 0xFULL);
    }
    return index;
}

std::size_t NTupleNetwork::relative_index_of(
    std::uint64_t codes, const std::vector<std::uint8_t>& shifts) const noexcept {
    // Same ordering rule as index_of, base 9 instead of 16 because a cell has
    // nine rank-relative codes. Not a shift, so this costs a multiply per cell.
    std::size_t index = 0;
    for (const auto shift : shifts) {
        index = index * 9U + static_cast<std::size_t>((codes >> shift) & 0xFULL);
    }
    return index;
}

template <bool Concurrent>
double NTupleNetwork::value_impl(Board board) const noexcept {
    const auto read = [](const float& slot) noexcept {
        if constexpr (Concurrent) {
            return load_relaxed(slot);
        } else {
            return slot;
        }
    };
    const auto packed = indexed_packed(board);
    const auto base = stage_offset(board);
    double total = 0.0;
    for (const auto& tuple : tuples_) {
        for (const auto& shifts : tuple.ordering_shifts) {
            total += static_cast<double>(
                read(weights_[base + tuple.lut_offset + index_of(packed, shifts)]));
        }
    }
    if (global_features_) {
        total += static_cast<double>(
            read(weights_[base + global_offset_ + global_feature_index(board)]));
    }
    if (structural_features_) {
        total += static_cast<double>(
            read(weights_[base + structural_offset_ + structural_feature_index(board)]));
    }
    if (relative_bank_) {
        // Always from the CLAMPED board, never from indexed_packed(): the bank
        // computes its own normalisation, and reading an already-relative board
        // would normalise twice and collapse every code to the same value.
        const auto codes = relative_bank_codes(clamped_packed(board));
        for (const auto& tuple : tuples_) {
            for (const auto& shifts : tuple.ordering_shifts) {
                total += static_cast<double>(
                    read(weights_[base + tuple.relative_offset +
                                  relative_index_of(codes, shifts)]));
            }
        }
    }
    return total;
}

template <bool Concurrent>
void NTupleNetwork::update_impl(Board board, double per_weight_delta) noexcept {
    const auto add = [](float& slot, float amount) noexcept {
        if constexpr (Concurrent) {
            add_relaxed(slot, amount);
        } else {
            slot += amount;
        }
    };
    const auto packed = indexed_packed(board);
    const auto base = stage_offset(board);
    const auto delta = static_cast<float>(per_weight_delta);
    for (const auto& tuple : tuples_) {
        for (const auto& shifts : tuple.ordering_shifts) {
            add(weights_[base + tuple.lut_offset + index_of(packed, shifts)], delta);
        }
    }
    if (global_features_) {
        add(weights_[base + global_offset_ + global_feature_index(board)], delta);
    }
    if (structural_features_) {
        add(weights_[base + structural_offset_ + structural_feature_index(board)], delta);
    }
    if (relative_bank_) {
        const auto codes = relative_bank_codes(clamped_packed(board));
        for (const auto& tuple : tuples_) {
            for (const auto& shifts : tuple.ordering_shifts) {
                add(weights_[base + tuple.relative_offset + relative_index_of(codes, shifts)],
                    delta);
            }
        }
    }
}

double NTupleNetwork::value(Board board) const noexcept {
    return concurrent_ ? value_impl<true>(board) : value_impl<false>(board);
}

void NTupleNetwork::update(Board board, double per_weight_delta) noexcept {
    if (concurrent_) {
        update_impl<true>(board, per_weight_delta);
    } else {
        update_impl<false>(board, per_weight_delta);
    }
}

void NTupleNetwork::active_indices(Board board, std::vector<std::size_t>& out) const {
    const auto packed = indexed_packed(board);
    const auto base = stage_offset(board);
    out.clear();
    out.reserve(active_weight_count_);
    for (const auto& tuple : tuples_) {
        for (const auto& shifts : tuple.ordering_shifts) {
            out.push_back(base + tuple.lut_offset + index_of(packed, shifts));
        }
    }
    if (global_features_) {
        out.push_back(base + global_offset_ + global_feature_index(board));
    }
    if (structural_features_) {
        out.push_back(base + structural_offset_ + structural_feature_index(board));
    }
    if (relative_bank_) {
        const auto codes = relative_bank_codes(clamped_packed(board));
        for (const auto& tuple : tuples_) {
            for (const auto& shifts : tuple.ordering_shifts) {
                out.push_back(base + tuple.relative_offset + relative_index_of(codes, shifts));
            }
        }
    }
}

std::size_t NTupleNetwork::total_weight_count() const noexcept {
    return weights_.size();
}

void NTupleNetwork::save(const std::filesystem::path& path) const {
    // Write to a temporary then rename, so an interrupted save can never be
    // read back as a complete file (same discipline as tablebase::LayerStore).
    auto temporary = path;
    temporary += ".tmp";
    {
        std::ofstream stream(temporary, std::ios::binary | std::ios::trunc);
        if (!stream) {
            throw std::runtime_error("cannot open weight file for writing: " +
                                     temporary.string());
        }
        write_exact(stream, kMagic, sizeof(kMagic));
        // Single-stage networks keep writing version 1, so files produced now
        // remain readable by anything that predates staging — and so every
        // weight file already on disk stays valid.
        // Write the OLDEST version that can express this network, so files stay
        // readable by anything that predates a feature it does not use.
        const auto needs_flags = global_features_ || structural_features_ ||
                                 relative_bank_ ||
                                 indexing_ == IndexingMode::relative ||
                                 (stage_count_ > 1 && stage_base_exponent_ != 10);
        const auto version = needs_flags ? kFlaggedFormatVersion
                           : stage_count_ > 1 ? kStagedFormatVersion
                                              : kFormatVersion;
        write_exact(stream, &version, sizeof(version));
        if (version >= kStagedFormatVersion) {
            const auto stages = static_cast<std::uint32_t>(stage_count_);
            write_exact(stream, &stages, sizeof(stages));
        }
        if (version >= kFlaggedFormatVersion) {
            const std::uint32_t flags =
                (global_features_ ? kFlagGlobalFeatures : 0U) |
                (indexing_ == IndexingMode::relative ? kFlagRelativeIndexing : 0U) |
                (structural_features_ ? kFlagStructuralFeatures : 0U) |
                (relative_bank_ ? kFlagRelativeBank : 0U) |
                (static_cast<std::uint32_t>(stage_base_exponent_) << kStageBaseShift);
            write_exact(stream, &flags, sizeof(flags));
        }

        const auto tuple_count = static_cast<std::uint32_t>(specs_.size());
        write_exact(stream, &tuple_count, sizeof(tuple_count));
        for (const auto& spec : specs_) {
            const auto cell_count = static_cast<std::uint32_t>(spec.cells.size());
            write_exact(stream, &cell_count, sizeof(cell_count));
            write_exact(stream, spec.cells.data(), spec.cells.size());
        }

        const auto weight_count = static_cast<std::uint64_t>(weights_.size());
        write_exact(stream, &weight_count, sizeof(weight_count));
        write_exact(stream, weights_.data(), weights_.size() * sizeof(float));

        stream.flush();
        if (!stream) {
            throw std::runtime_error("failed writing weight file: " + temporary.string());
        }
    }
    std::filesystem::rename(temporary, path);
}

namespace {

// Reads and validates the header shared by load() and load_from(), returning
// the tuple definitions the file was written with. Leaves the stream
// positioned at the weight count.
struct HeaderInfo {
    std::vector<TupleSpec> specs;
    std::size_t stage_count{1};
    bool global_features{false};
    IndexingMode indexing{IndexingMode::absolute};
    std::uint8_t stage_base{10};
    bool structural_features{false};
    bool relative_bank{false};
};

[[nodiscard]] HeaderInfo read_header_specs(
    std::ifstream& stream, const std::filesystem::path& path) {
    std::array<char, sizeof(kMagic)> magic{};
    read_exact(stream, magic.data(), magic.size(), path);
    if (std::memcmp(magic.data(), kMagic, sizeof(kMagic)) != 0) {
        throw std::runtime_error("not an n-tuple weight file: " + path.string());
    }
    std::uint32_t version = 0;
    read_exact(stream, &version, sizeof(version), path);
    if (version != kFormatVersion && version != kStagedFormatVersion &&
        version != kFlaggedFormatVersion) {
        throw std::runtime_error("unsupported weight file version " +
                                 std::to_string(version) + " in " + path.string());
    }

    HeaderInfo info;
    if (version >= kStagedFormatVersion) {
        std::uint32_t stages = 0;
        read_exact(stream, &stages, sizeof(stages), path);
        if (stages == 0) {
            throw std::runtime_error("weight file declares zero stages: " + path.string());
        }
        info.stage_count = stages;
    }
    if (version >= kFlaggedFormatVersion) {
        std::uint32_t flags = 0;
        read_exact(stream, &flags, sizeof(flags), path);
        info.global_features = (flags & kFlagGlobalFeatures) != 0U;
        info.indexing = (flags & kFlagRelativeIndexing) != 0U ? IndexingMode::relative
                                                             : IndexingMode::absolute;
        info.structural_features = (flags & kFlagStructuralFeatures) != 0U;
        info.relative_bank = (flags & kFlagRelativeBank) != 0U;
        const auto recorded = static_cast<std::uint8_t>((flags >> kStageBaseShift) & 0xFFU);
        info.stage_base = recorded == 0 ? 10 : recorded;
    }

    std::uint32_t tuple_count = 0;
    read_exact(stream, &tuple_count, sizeof(tuple_count), path);
    info.specs.resize(tuple_count);
    for (std::uint32_t index = 0; index < tuple_count; ++index) {
        std::uint32_t cell_count = 0;
        read_exact(stream, &cell_count, sizeof(cell_count), path);
        info.specs[index].cells.resize(cell_count);
        if (cell_count != 0) {
            read_exact(stream, info.specs[index].cells.data(), cell_count, path);
        }
    }
    return info;
}

}  // namespace

NTupleNetwork NTupleNetwork::load_from(const std::filesystem::path& path) {
    std::ifstream stream(path, std::ios::binary);
    if (!stream) {
        throw std::runtime_error("cannot open weight file: " + path.string());
    }
    // The NTupleNetwork constructor validates the specs themselves (cell
    // ranges, tuple sizes), so a corrupt header cannot produce a live network.
    auto header = read_header_specs(stream, path);
    NTupleNetwork network(std::move(header.specs), header.stage_count,
                          header.global_features, header.indexing, header.stage_base,
                          header.structural_features, header.relative_bank);

    std::uint64_t weight_count = 0;
    read_exact(stream, &weight_count, sizeof(weight_count), path);
    if (weight_count != network.weights_.size()) {
        throw std::runtime_error("weight file declares " + std::to_string(weight_count) +
                                 " weights but its tuples need " +
                                 std::to_string(network.weights_.size()) + ": " + path.string());
    }
    read_exact(stream, network.weights_.data(), network.weights_.size() * sizeof(float), path);
    stream.peek();
    if (!stream.eof()) {
        throw std::runtime_error("weight file has unexpected trailing data: " + path.string());
    }
    return network;
}

void NTupleNetwork::load(const std::filesystem::path& path) {
    std::ifstream stream(path, std::ios::binary);
    if (!stream) {
        throw std::runtime_error("cannot open weight file: " + path.string());
    }

    std::array<char, sizeof(kMagic)> magic{};
    read_exact(stream, magic.data(), magic.size(), path);
    if (std::memcmp(magic.data(), kMagic, sizeof(kMagic)) != 0) {
        throw std::runtime_error("not an n-tuple weight file: " + path.string());
    }
    std::uint32_t version = 0;
    read_exact(stream, &version, sizeof(version), path);
    if (version != kFormatVersion && version != kStagedFormatVersion &&
        version != kFlaggedFormatVersion) {
        throw std::runtime_error("unsupported weight file version " +
                                 std::to_string(version) + " in " + path.string());
    }
    // Stage count is part of the shape: loading a 4-stage file into a
    // single-stage network would read the wrong table for most boards while
    // still producing finite, plausible numbers.
    std::size_t file_stages = 1;
    if (version >= kStagedFormatVersion) {
        std::uint32_t stages = 0;
        read_exact(stream, &stages, sizeof(stages), path);
        file_stages = stages;
    }
    bool file_global = false;
    bool file_relative_bank = false;
    bool file_global_flags_read = false;
    auto file_indexing_read = IndexingMode::absolute;
    if (version >= kFlaggedFormatVersion) {
        std::uint32_t flags = 0;
        read_exact(stream, &flags, sizeof(flags), path);
        file_global = (flags & kFlagGlobalFeatures) != 0U;
        file_relative_bank = (flags & kFlagRelativeBank) != 0U;
        file_indexing_read = (flags & kFlagRelativeIndexing) != 0U ? IndexingMode::relative
                                                                  : IndexingMode::absolute;
        file_global_flags_read = true;
    }
    const auto file_indexing = (version >= kFlaggedFormatVersion && file_global_flags_read)
        ? file_indexing_read
        : IndexingMode::absolute;
    if (file_indexing != indexing_) {
        throw std::runtime_error("weight file indexing mode does not match: " + path.string());
    }
    if (file_global != global_features_) {
        throw std::runtime_error("weight file global-feature setting does not match: " +
                                 path.string());
    }
    if (file_relative_bank != relative_bank_) {
        throw std::runtime_error("weight file relative-bank setting does not match: " +
                                 path.string());
    }
    if (file_stages != stage_count_) {
        throw std::runtime_error("weight file has " + std::to_string(file_stages) +
                                 " stages but this network has " +
                                 std::to_string(stage_count_) + ": " + path.string());
    }

    // The tuple definitions are validated against this network's own shape.
    // Loading weights trained for different patterns would otherwise produce
    // plausible-looking nonsense rather than an error.
    std::uint32_t tuple_count = 0;
    read_exact(stream, &tuple_count, sizeof(tuple_count), path);
    if (tuple_count != specs_.size()) {
        throw std::runtime_error("weight file has " + std::to_string(tuple_count) +
                                 " tuples but this network has " +
                                 std::to_string(specs_.size()) + ": " + path.string());
    }
    for (std::size_t index = 0; index < specs_.size(); ++index) {
        std::uint32_t cell_count = 0;
        read_exact(stream, &cell_count, sizeof(cell_count), path);
        if (cell_count != specs_[index].cells.size()) {
            throw std::runtime_error("weight file tuple " + std::to_string(index) +
                                     " has a different cell count: " + path.string());
        }
        std::vector<std::uint8_t> cells(cell_count);
        read_exact(stream, cells.data(), cells.size(), path);
        if (cells != specs_[index].cells) {
            throw std::runtime_error("weight file tuple " + std::to_string(index) +
                                     " has different cells: " + path.string());
        }
    }

    std::uint64_t weight_count = 0;
    read_exact(stream, &weight_count, sizeof(weight_count), path);
    if (weight_count != weights_.size()) {
        throw std::runtime_error("weight file has " + std::to_string(weight_count) +
                                 " weights but this network needs " +
                                 std::to_string(weights_.size()) + ": " + path.string());
    }
    read_exact(stream, weights_.data(), weights_.size() * sizeof(float), path);

    // Reject trailing garbage: a file longer than its declared contents means
    // it was produced by something we do not understand.
    stream.peek();
    if (!stream.eof()) {
        throw std::runtime_error("weight file has unexpected trailing data: " + path.string());
    }
}

std::string NTupleNetwork::fingerprint() const {
    // FNV-1a over the raw weight bytes, plus the network shape.
    std::uint64_t hash = 0xCBF29CE484222325ULL;
    const auto* bytes = reinterpret_cast<const unsigned char*>(weights_.data());
    const auto byte_count = weights_.size() * sizeof(float);
    for (std::size_t index = 0; index < byte_count; ++index) {
        hash ^= bytes[index];
        hash *= 0x100000001B3ULL;
    }

    std::ostringstream out;
    out << "tuples=";
    for (std::size_t index = 0; index < specs_.size(); ++index) {
        out << (index == 0 ? "" : "+") << specs_[index].cells.size();
    }
    out << ",active=" << active_weight_count_;
    if (global_features_) {
        out << ",global";
    }
    if (structural_features_) {
        out << ",structural";
    }
    if (relative_bank_) {
        out << ",relbank";
    }
    if (indexing_ == IndexingMode::relative) {
        out << ",relative";
    }
    if (stage_count_ > 1) {
        out << ",stagebase=" << static_cast<int>(stage_base_exponent_);
    }
    // Only emitted when staged, so single-stage fingerprints stay byte-for-byte
    // what they were before staging existed and old experiment records remain
    // comparable to new ones.
    if (stage_count_ > 1) {
        out << ",stages=" << stage_count_;
    }
    out << ",weights=" << weights_.size()
        << ",hash=" << std::hex << hash;
    return out.str();
}

}  // namespace adversarial_2048::learning
