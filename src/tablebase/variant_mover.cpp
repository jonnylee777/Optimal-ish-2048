#include "tablebase/variant_mover.hpp"

#include "tablebase/formation.hpp"

#include <array>

namespace adversarial_2048::tablebase {
namespace {

constexpr std::uint8_t kMaxMergeableExponent = 0xE;

struct LineResult {
    std::uint16_t line{};
    std::uint64_t score{};
};

// Compacts and merges one wall-delimited segment [begin, end) of a line.
//
// Tiles are gathered starting from the edge they are moving TOWARD, so the
// pair nearest that edge merges first — this is what 2048 does, and getting it
// backwards silently mis-scores rows like [2,2,4,4] (right => 4,8 not 8,4).
void merge_segment(
    const std::array<std::uint8_t, kBoardWidth>& cells,
    std::size_t begin, std::size_t end,  // half-open
    bool leftward,
    std::array<std::uint8_t, kBoardWidth>& out, std::uint64_t& score) noexcept {
    std::array<std::uint8_t, kBoardWidth> gathered{};
    std::size_t count = 0;
    if (leftward) {
        for (std::size_t index = begin; index < end; ++index) {
            if (cells[index] != 0) {
                gathered[count++] = cells[index];
            }
        }
    } else {
        for (std::size_t index = end; index-- > begin;) {
            if (cells[index] != 0) {
                gathered[count++] = cells[index];
            }
        }
    }

    const int step = leftward ? +1 : -1;
    auto out_index = static_cast<int>(leftward ? begin : end - 1);
    for (std::size_t index = 0; index < count;) {
        auto exponent = gathered[index];
        if (index + 1 < count && gathered[index + 1] == exponent &&
            exponent < kMaxMergeableExponent) {
            ++exponent;
            score += std::uint64_t{1} << exponent;
            index += 2;
        } else {
            ++index;
        }
        out[static_cast<std::size_t>(out_index)] = exponent;
        out_index += step;
    }
}

[[nodiscard]] LineResult move_line(std::uint16_t line, bool leftward) noexcept {
    std::array<std::uint8_t, kBoardWidth> cells{};
    for (std::size_t index = 0; index < kBoardWidth; ++index) {
        cells[index] = static_cast<std::uint8_t>((line >> (index * 4U)) & 0xFU);
    }

    std::array<std::uint8_t, kBoardWidth> out{};
    std::uint64_t score = 0;

    // Walls stay exactly where they are and delimit segments.
    std::size_t segment_begin = 0;
    for (std::size_t index = 0; index <= kBoardWidth; ++index) {
        const bool is_wall = index < kBoardWidth && cells[index] == kWallExponent;
        if (index == kBoardWidth || is_wall) {
            if (index > segment_begin) {
                merge_segment(cells, segment_begin, index, leftward, out, score);
            }
            if (is_wall) {
                out[index] = kWallExponent;
            }
            segment_begin = index + 1;
        }
    }

    std::uint16_t moved_line = 0;
    for (std::size_t index = 0; index < kBoardWidth; ++index) {
        moved_line |= static_cast<std::uint16_t>(
            static_cast<std::uint16_t>(out[index]) << (index * 4U));
    }
    return LineResult{moved_line, score};
}

[[nodiscard]] VariantMoveResult move_horizontal(
    std::uint64_t packed, bool leftward) noexcept {
    std::uint64_t result = 0;
    std::uint64_t score = 0;
    for (std::size_t row = 0; row < kBoardWidth; ++row) {
        const auto shift = row * 16U;
        const auto line = static_cast<std::uint16_t>((packed >> shift) & 0xFFFFU);
        const auto moved = move_line(line, leftward);
        result |= static_cast<std::uint64_t>(moved.line) << shift;
        score += moved.score;
    }
    return VariantMoveResult{result, score, result != packed};
}

}  // namespace

VariantMoveResult move_variant(std::uint64_t packed, Direction direction) noexcept {
    // Vertical moves transpose so columns read as rows, move horizontally, then
    // transpose back. Walls transpose along with everything else, so the
    // wall-aware line mover stays correct.
    switch (direction) {
        case Direction::left:
            return move_horizontal(packed, true);
        case Direction::right:
            return move_horizontal(packed, false);
        case Direction::up:
        case Direction::down: {
            const auto transposed = transpose(Board{packed, 0}).packed_exponents;
            const auto moved = move_horizontal(transposed, direction == Direction::up);
            const auto restored = transpose(Board{moved.packed, 0}).packed_exponents;
            return VariantMoveResult{restored, moved.score, restored != packed};
        }
    }
    return VariantMoveResult{packed, 0, false};
}

bool variant_has_move(std::uint64_t packed) noexcept {
    for (const auto direction : kDirections) {
        if (move_variant(packed, direction).moved) {
            return true;
        }
    }
    return false;
}

}  // namespace adversarial_2048::tablebase
