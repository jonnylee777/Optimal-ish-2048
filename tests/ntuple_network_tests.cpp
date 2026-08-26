#include "core/board.hpp"
#include "core/spawn.hpp"
#include "learning/ntuple_network.hpp"
#include "tablebase/formation.hpp"  // apply_symmetry: separately tested; used here
                                    // only to verify N1's invariance claim.

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <unistd.h>
#include <utility>
#include <vector>

namespace a2048 = adversarial_2048;
namespace tb = adversarial_2048::tablebase;
namespace nn = adversarial_2048::learning;

namespace {

class TestFailure : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

#define CHECK(condition)                                                                          \
    do {                                                                                          \
        if (!(condition)) {                                                                       \
            throw TestFailure(std::string("check failed: ") + #condition + " at line " +         \
                              std::to_string(__LINE__));                                           \
        }                                                                                         \
    } while (false)

#define CHECK_THROWS(expression)                                                                  \
    do {                                                                                          \
        bool threw = false;                                                                       \
        try {                                                                                     \
            (void)(expression);                                                                    \
        } catch (const std::exception&) {                                                          \
            threw = true;                                                                          \
        }                                                                                          \
        if (!threw) {                                                                              \
            throw TestFailure(std::string("expected throw: ") + #expression + " at line " +       \
                              std::to_string(__LINE__));                                           \
        }                                                                                          \
    } while (false)

using Cells = a2048::CellArray;

class TemporaryDirectory {
public:
    explicit TemporaryDirectory(const std::string& tag) {
        path_ = std::filesystem::temp_directory_path() /
                ("a2048_ntuple_" + tag + "_" + std::to_string(::getpid()));
        std::filesystem::remove_all(path_);
        std::filesystem::create_directories(path_);
    }
    ~TemporaryDirectory() {
        std::error_code error;
        std::filesystem::remove_all(path_, error);
    }
    TemporaryDirectory(const TemporaryDirectory&) = delete;
    TemporaryDirectory& operator=(const TemporaryDirectory&) = delete;
    [[nodiscard]] const std::filesystem::path& path() const { return path_; }

private:
    std::filesystem::path path_;
};

// --- independent transcription of the reference indexing scheme ---
// Written from the reference formula `address = address * numTileValues + value`
// (wjaskowski/mastering-2048, Tiling2048.java) operating on a decoded cell
// array, deliberately NOT calling into ntuple_network.cpp, so agreement is a
// real cross-check rather than a tautology.
[[nodiscard]] std::size_t reference_index(
    const Cells& cells, const std::vector<std::uint8_t>& tuple_cells) {
    std::size_t address = 0;
    for (const auto cell : tuple_cells) {
        std::size_t value = cells[cell];
        if (value > 15) {
            value = 15;  // 4 bits per cell; the extension plane clamps
        }
        address = address * 16U + value;
    }
    return address;
}

// The 8 dihedral cell maps, transcribed independently of the production code.
[[nodiscard]] std::vector<std::vector<std::uint8_t>> reference_orderings(
    const std::vector<std::uint8_t>& base) {
    const auto at = [](std::size_t r, std::size_t c) {
        return static_cast<std::uint8_t>(r * 4 + c);
    };
    std::vector<std::vector<std::uint8_t>> result;
    for (int symmetry = 0; symmetry < 8; ++symmetry) {
        std::vector<std::uint8_t> mapped;
        for (const auto cell : base) {
            const std::size_t r = cell / 4;
            const std::size_t c = cell % 4;
            std::uint8_t out = 0;
            switch (symmetry) {
                case 0: out = at(r, c); break;
                case 1: out = at(c, 3 - r); break;
                case 2: out = at(3 - r, 3 - c); break;
                case 3: out = at(3 - c, r); break;
                case 4: out = at(r, 3 - c); break;
                case 5: out = at(3 - r, c); break;
                case 6: out = at(c, r); break;
                default: out = at(3 - c, 3 - r); break;
            }
            mapped.push_back(out);
        }
        if (std::find(result.begin(), result.end(), mapped) == result.end()) {
            result.push_back(mapped);
        }
    }
    return result;
}

// Reference value: sum over tuples and their distinct orderings.
[[nodiscard]] double reference_value(
    const Cells& cells, const std::vector<nn::TupleSpec>& specs,
    const std::vector<std::vector<float>>& luts) {
    double total = 0.0;
    for (std::size_t index = 0; index < specs.size(); ++index) {
        for (const auto& ordering : reference_orderings(specs[index].cells)) {
            total += static_cast<double>(luts[index][reference_index(cells, ordering)]);
        }
    }
    return total;
}

// Extracts the per-tuple LUTs from a network by probing it: set one weight at
// a time is impractical, so instead we rebuild the LUTs by asking the network
// for values on boards that activate exactly one index. Simpler and more
// robust: use update() to write a known pattern, then compare against a
// reference computed on the same pattern.
void test_gate_index_and_value_match_reference() {
    const auto specs = nn::default_tuple_specs();
    nn::NTupleNetwork network(specs);

    // Give every active weight for a set of seed boards a distinct value by
    // updating with those boards, then compare network.value() against the
    // independent reference over fresh random boards.
    std::mt19937_64 rng(20260825);
    std::uniform_int_distribution<int> exponent(0, 15);

    // Mirror the network's weights in plain per-tuple vectors, applying the
    // same updates through the reference path.
    std::vector<std::vector<float>> luts;
    for (const auto& spec : specs) {
        std::size_t size = 1;
        for (std::size_t i = 0; i < spec.cells.size(); ++i) {
            size *= 16U;
        }
        luts.emplace_back(size, 0.0F);
    }

    for (int trial = 0; trial < 200; ++trial) {
        Cells cells{};
        for (auto& cell : cells) {
            cell = static_cast<std::uint8_t>(exponent(rng));
        }
        const auto board = a2048::encode(cells);
        const auto delta = 0.5 + 0.25 * static_cast<double>(trial);

        network.update(board, delta);
        // Same update through the reference: every distinct ordering of every
        // tuple gets += delta.
        for (std::size_t index = 0; index < specs.size(); ++index) {
            for (const auto& ordering : reference_orderings(specs[index].cells)) {
                luts[index][reference_index(cells, ordering)] += static_cast<float>(delta);
            }
        }
    }

    std::size_t compared = 0;
    for (int trial = 0; trial < 3000; ++trial) {
        Cells cells{};
        for (auto& cell : cells) {
            cell = static_cast<std::uint8_t>(exponent(rng));
        }
        const auto board = a2048::encode(cells);
        const auto expected = reference_value(cells, specs, luts);
        const auto actual = network.value(board);
        // Both accumulate float32 weights into a double in the same order, so
        // this should be exact.
        CHECK(actual == expected);
        ++compared;
    }
    std::cout << "        (cross-checked " << compared << " boards)\n";
}

void test_default_network_shape_and_size() {
    nn::NTupleNetwork network(nn::default_tuple_specs());
    // 2 straight 4-tuples + 2 rectangles of 6: 2*16^4 + 2*16^6.
    const std::size_t expected = 2U * 65536U + 2U * 16777216U;
    CHECK(network.total_weight_count() == expected);
    CHECK(network.total_weight_count() == 33685504U);
    // 4 tuples, 8 distinct orderings each for these asymmetric shapes.
    CHECK(network.active_weight_count() == 32U);
}

void test_ordering_dedup_matches_reference() {
    // Dedup is on the ORDERED cell list, so it triggers less often than one
    // might expect. A corner 2x2 square still yields all 8 distinct
    // orderings (transpose reorders the same four cells), which matches the
    // reference implementation's behaviour.
    nn::NTupleNetwork square({nn::TupleSpec{{0, 1, 4, 5}}});
    CHECK(square.active_weight_count() == reference_orderings({0, 1, 4, 5}).size());
    CHECK(square.active_weight_count() == 8U);

    // A single cell is where dedup actually bites: a corner has only 4
    // distinct images, and a length-1 ordering carries no order information.
    nn::NTupleNetwork corner({nn::TupleSpec{{0}}});
    CHECK(corner.active_weight_count() == reference_orderings({0}).size());
    CHECK(corner.active_weight_count() == 4U);

    nn::NTupleNetwork inner({nn::TupleSpec{{5}}});
    CHECK(inner.active_weight_count() == reference_orderings({5}).size());
    CHECK(inner.active_weight_count() == 4U);
}

// Returns true when every distinct ordering of every tuple lands on a
// different LUT index for this board (computed via the independent reference).
[[nodiscard]] bool all_orderings_distinct(
    const Cells& cells, const std::vector<nn::TupleSpec>& specs) {
    for (const auto& spec : specs) {
        std::vector<std::size_t> indices;
        for (const auto& ordering : reference_orderings(spec.cells)) {
            indices.push_back(reference_index(cells, ordering));
        }
        std::sort(indices.begin(), indices.end());
        if (std::adjacent_find(indices.begin(), indices.end()) != indices.end()) {
            return false;
        }
    }
    return true;
}

void test_update_is_linear_in_delta() {
    // The TD update relies on value() responding linearly to a weight update.
    const auto specs = nn::default_tuple_specs();
    Cells cells{};
    cells[0] = 1;
    cells[1] = 2;
    cells[2] = 3;
    cells[3] = 4;
    cells[5] = 5;
    const auto board = a2048::encode(cells);

    nn::NTupleNetwork single(specs);
    single.update(board, 1.0);
    const auto one = single.value(board);

    nn::NTupleNetwork doubled(specs);
    doubled.update(board, 2.0);
    const auto two = doubled.value(board);

    CHECK(one > 0.0);
    CHECK(std::abs(two - 2.0 * one) < 1e-9);
}

void test_sparse_boards_collide_and_overshoot() {
    // IMPORTANT property for the TD trainer, not just trivia.
    //
    // Different symmetric orderings of a tuple can land on the SAME LUT index
    // — most commonly on sparse boards, where several orderings map entirely
    // into the empty region and all index 0. When that happens, one update of
    // `d` per active weight moves value() by MORE than active_count * d,
    // because the colliding weight is both incremented and read multiple
    // times.
    //
    // Consequence: the papers' claim that alpha = 1.0 "immediately reduces
    // the prediction error to zero" holds only when orderings are distinct.
    // On sparse (early-game) boards alpha = 1.0 overshoots. That is the
    // reason the trainer exposes alpha as a tunable rather than hardcoding 1.
    const auto specs = nn::default_tuple_specs();

    Cells sparse{};
    sparse[0] = 1;
    sparse[1] = 2;
    sparse[2] = 3;
    sparse[3] = 4;
    sparse[5] = 5;
    CHECK(!all_orderings_distinct(sparse, specs));  // collisions confirmed

    nn::NTupleNetwork network(specs);
    const double delta = 0.125;
    const auto board = a2048::encode(sparse);
    network.update(board, delta);
    const auto naive = static_cast<double>(network.active_weight_count()) * delta;
    CHECK(network.value(board) > naive);  // strictly overshoots

    // A densely filled board with distinct values has no collisions, and
    // there the naive identity holds exactly.
    Cells dense{};
    for (std::size_t index = 0; index < a2048::kCellCount; ++index) {
        dense[index] = static_cast<std::uint8_t>(1 + index % 15);
    }
    if (all_orderings_distinct(dense, specs)) {
        nn::NTupleNetwork exact(specs);
        const auto dense_board = a2048::encode(dense);
        exact.update(dense_board, delta);
        const auto expected =
            static_cast<double>(exact.active_weight_count()) * delta;
        CHECK(std::abs(exact.value(dense_board) - expected) < 1e-9);
    }
}

void test_value_is_zero_on_fresh_network() {
    nn::NTupleNetwork network(nn::default_tuple_specs());
    Cells cells{};
    cells[0] = 3;
    cells[5] = 7;
    CHECK(network.value(a2048::encode(cells)) == 0.0);
}

void test_gate_rotation_invariance_empirically() {
    // Symmetric weight sharing should make the value invariant under all 8
    // dihedral transforms. Verify rather than assume, as H5 does.
    nn::NTupleNetwork network(nn::default_tuple_specs());

    std::mt19937_64 seed_rng(4242);
    std::uniform_int_distribution<int> exponent(0, 15);
    // Train a little structure into the weights first; an all-zero network
    // would pass trivially.
    for (int trial = 0; trial < 300; ++trial) {
        Cells cells{};
        for (auto& cell : cells) {
            cell = static_cast<std::uint8_t>(exponent(seed_rng));
        }
        network.update(a2048::encode(cells), 1.0 + 0.01 * static_cast<double>(trial));
    }

    const std::array<tb::Symmetry, 8> all{
        tb::Symmetry::identity,       tb::Symmetry::reverse_lr,
        tb::Symmetry::reverse_ud,     tb::Symmetry::transpose_main,
        tb::Symmetry::transpose_anti, tb::Symmetry::rotate_180,
        tb::Symmetry::rotate_left,    tb::Symmetry::rotate_right};

    std::mt19937_64 rng(99);
    std::size_t checks = 0;
    for (int trial = 0; trial < 500; ++trial) {
        Cells cells{};
        for (auto& cell : cells) {
            cell = static_cast<std::uint8_t>(exponent(rng));
        }
        const auto packed = a2048::encode(cells).packed_exponents;
        const auto reference = network.value(a2048::Board{packed, 0});
        for (const auto symmetry : all) {
            const auto moved = tb::apply_symmetry(packed, symmetry);
            CHECK(std::abs(network.value(a2048::Board{moved, 0}) - reference) < 1e-6);
            ++checks;
        }
    }
    std::cout << "        (" << checks << " invariance checks)\n";
}

void test_extension_plane_clamps_to_fifteen() {
    nn::NTupleNetwork network(nn::default_tuple_specs());

    Cells fifteens{};
    fifteens[0] = 15;
    fifteens[1] = 4;
    const auto plain = a2048::encode(fifteens);
    CHECK(plain.exponent_high_bits == 0);

    // Same board but cell 0 holds exponent 16 (a 65536 tile), which sets the
    // extension plane. It must index identically to exponent 15.
    auto extended = a2048::with_cell(plain, 0, 16);
    CHECK(extended.exponent_high_bits != 0);

    network.update(plain, 1.0);
    CHECK(std::abs(network.value(extended) - network.value(plain)) < 1e-12);

    // And exponent 31 clamps the same way.
    auto very_extended = a2048::with_cell(plain, 0, 31);
    CHECK(very_extended.exponent_high_bits != 0);
    CHECK(std::abs(network.value(very_extended) - network.value(plain)) < 1e-12);
}

void test_save_load_round_trips_bit_exactly() {
    const TemporaryDirectory directory("roundtrip");
    const auto path = directory.path() / "weights.bin";

    nn::NTupleNetwork original(nn::default_tuple_specs());
    std::mt19937_64 rng(7);
    std::uniform_int_distribution<int> exponent(0, 15);
    for (int trial = 0; trial < 500; ++trial) {
        Cells cells{};
        for (auto& cell : cells) {
            cell = static_cast<std::uint8_t>(exponent(rng));
        }
        original.update(a2048::encode(cells), 1.0 / 3.0);
    }
    original.save(path);

    nn::NTupleNetwork restored(nn::default_tuple_specs());
    restored.load(path);
    CHECK(restored.fingerprint() == original.fingerprint());

    // Bit-exact values on fresh boards.
    std::mt19937_64 check_rng(8);
    for (int trial = 0; trial < 500; ++trial) {
        Cells cells{};
        for (auto& cell : cells) {
            cell = static_cast<std::uint8_t>(exponent(check_rng));
        }
        const auto board = a2048::encode(cells);
        CHECK(restored.value(board) == original.value(board));
    }

    // No stray temporary left behind.
    for (const auto& entry : std::filesystem::directory_iterator(directory.path())) {
        CHECK(entry.path().extension() != ".tmp");
    }
}

void test_load_rejects_mismatched_and_corrupt_files() {
    const TemporaryDirectory directory("reject");
    const auto path = directory.path() / "weights.bin";

    nn::NTupleNetwork small({nn::TupleSpec{{0, 1, 2, 3}}});
    small.save(path);

    // Different tuple shape must be refused, not silently misread.
    nn::NTupleNetwork different({nn::TupleSpec{{4, 5, 6, 7}}});
    CHECK_THROWS(different.load(path));

    // Different tuple count.
    nn::NTupleNetwork two_tuples({nn::TupleSpec{{0, 1, 2, 3}}, nn::TupleSpec{{4, 5, 6, 7}}});
    CHECK_THROWS(two_tuples.load(path));

    // Truncated file.
    const auto truncated = directory.path() / "truncated.bin";
    {
        std::ifstream source(path, std::ios::binary);
        std::vector<char> bytes((std::istreambuf_iterator<char>(source)),
                                std::istreambuf_iterator<char>());
        bytes.resize(bytes.size() / 2);
        std::ofstream out(truncated, std::ios::binary);
        out.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
    }
    nn::NTupleNetwork same_shape({nn::TupleSpec{{0, 1, 2, 3}}});
    CHECK_THROWS(same_shape.load(truncated));

    // Not a weight file at all.
    const auto garbage = directory.path() / "garbage.bin";
    {
        std::ofstream out(garbage, std::ios::binary);
        out << "definitely not an n-tuple weight file, not even close";
    }
    CHECK_THROWS(same_shape.load(garbage));

    // Missing file.
    CHECK_THROWS(same_shape.load(directory.path() / "does_not_exist.bin"));
}

void test_invalid_specs_are_rejected() {
    CHECK_THROWS(nn::NTupleNetwork({}));
    CHECK_THROWS(nn::NTupleNetwork({nn::TupleSpec{{}}}));
    CHECK_THROWS(nn::NTupleNetwork({nn::TupleSpec{{0, 1, 16}}}));  // cell out of range
    CHECK_THROWS(nn::NTupleNetwork({nn::TupleSpec{{0, 1, 2, 3, 4, 5, 6, 7, 8}}}));  // too big
}

// load_from() must rebuild a network from the file alone, including a shape
// the caller never mentions. Uses a deliberately non-default shape: if it
// silently fell back to default_tuple_specs() the weights would still load and
// the values would be wrong but plausible.
void test_load_from_reconstructs_shape_from_file() {
    const std::vector<nn::TupleSpec> custom{
        nn::TupleSpec{{0, 1, 5}},
        nn::TupleSpec{{2, 3, 6, 7}},
    };
    nn::NTupleNetwork original(custom);
    const auto board = a2048::encode(a2048::CellArray{1, 2, 3, 4, 5, 6, 7, 8,
                                                     9, 10, 11, 12, 13, 14, 15, 0});
    original.update(board, 2.5);
    const auto expected = original.value(board);

    const auto path = std::filesystem::temp_directory_path() / "a2048_load_from_test.bin";
    original.save(path);
    const auto restored = nn::NTupleNetwork::load_from(path);
    std::filesystem::remove(path);

    CHECK(restored.specs().size() == custom.size());
    CHECK(restored.specs()[0].cells == custom[0].cells);
    CHECK(restored.specs()[1].cells == custom[1].cells);
    CHECK(restored.total_weight_count() == original.total_weight_count());
    CHECK(std::abs(restored.value(board) - expected) < 1e-9);
    CHECK(restored.fingerprint() == original.fingerprint());
}

void test_load_from_rejects_corrupt_files() {
    const auto path = std::filesystem::temp_directory_path() / "a2048_load_from_bad.bin";
    {
        std::ofstream stream(path, std::ios::binary);
        stream << "not a weight file at all";
    }
    CHECK_THROWS(static_cast<void>(nn::NTupleNetwork::load_from(path)));
    std::filesystem::remove(path);
}

// A mistyped configuration name must fail loudly. Falling back to a default
// would train a different network than the operator asked for and label the
// result with the name they typed.
void test_named_tuple_specs_rejects_unknown_names() {
    CHECK_THROWS(static_cast<void>(nn::named_tuple_specs("larg")));
    for (const auto& name : nn::tuple_configuration_names()) {
        CHECK(!nn::named_tuple_specs(name).empty());
    }
}

// Pin the advertised sizes: "large" is described as ~320 MB, and a silent
// change to the shapes would invalidate the memory budgeting this project
// depends on. Checked arithmetically, without allocating the network.
void test_named_tuple_specs_have_expected_sizes() {
    const auto weights_for = [](const std::vector<nn::TupleSpec>& specs) {
        std::size_t total = 0;
        for (const auto& spec : specs) {
            std::size_t size = 1;
            for (std::size_t index = 0; index < spec.cells.size(); ++index) {
                size *= 16U;
            }
            total += size;
        }
        return total;
    };
    CHECK(weights_for(nn::named_tuple_specs("default")) == 33685504U);
    CHECK(weights_for(nn::named_tuple_specs("large")) == 83886080U);

    // Every "large" tuple must be a distinct 6-cell set with in-range,
    // non-repeating cells — a duplicated cell would quietly waste a whole
    // 16^6 table on a lower-order pattern.
    const auto large = nn::named_tuple_specs("large");
    CHECK(large.size() == 5);
    for (const auto& spec : large) {
        CHECK(spec.cells.size() == 6);
        auto sorted = spec.cells;
        std::sort(sorted.begin(), sorted.end());
        CHECK(std::unique(sorted.begin(), sorted.end()) == sorted.end());
        CHECK(sorted.back() < 16);
    }
}

// stage_of() must be MONOTONE NON-DECREASING over any real game. If a board
// could move backwards through stages, a game would revisit an earlier weight
// set carrying values learned under a different distribution.
void test_stage_function_is_monotone_over_a_game() {
    constexpr std::size_t kStages = 4;
    std::mt19937_64 rng(0x5747E5EED);
    for (int game = 0; game < 40; ++game) {
        a2048::Board board{};
        static_cast<void>(a2048::spawn_random(board, rng));
        static_cast<void>(a2048::spawn_random(board, rng));
        std::size_t previous = nn::stage_of(board, kStages);
        for (int step = 0; step < 400; ++step) {
            const auto direction = a2048::kDirections[rng() % 4];
            const auto result = a2048::move(board, direction);
            if (!result.moved) {
                continue;
            }
            board = result.board;
            const auto current = nn::stage_of(board, kStages);
            CHECK(current >= previous);
            previous = current;
            if (!a2048::spawn_random(board, rng).has_value()) {
                break;
            }
        }
    }
}

// A single-stage network must be completely unaffected by the staging code —
// this is what lets every existing weight file and result stay valid.
void test_single_stage_is_unchanged() {
    const auto board = a2048::encode(a2048::CellArray{11, 2, 3, 4, 5, 6, 7, 8,
                                                     9, 10, 1, 12, 13, 14, 15, 0});
    CHECK(nn::stage_of(board, 1) == 0);

    nn::NTupleNetwork one(nn::default_tuple_specs());
    nn::NTupleNetwork defaulted(nn::default_tuple_specs(), 1);
    one.update(board, 1.5);
    defaulted.update(board, 1.5);
    CHECK(one.total_weight_count() == defaulted.total_weight_count());
    CHECK(one.fingerprint() == defaulted.fingerprint());
}

// Different stages must be genuinely independent: updating a board in one
// stage must not move the value of a board in another.
void test_stages_are_independent() {
    constexpr std::size_t kStages = 4;
    nn::NTupleNetwork network(nn::default_tuple_specs(), kStages);
    CHECK(network.total_weight_count() ==
          nn::NTupleNetwork(nn::default_tuple_specs()).total_weight_count() * kStages);

    // Same tile pattern in the low cells, but different max tiles, so the two
    // boards land in different stages.
    const auto early = a2048::encode(a2048::CellArray{1, 2, 3, 4, 0, 0, 0, 0,
                                                     0, 0, 0, 0, 0, 0, 0, 0});
    const auto late = a2048::encode(a2048::CellArray{1, 2, 3, 4, 0, 0, 0, 0,
                                                    0, 0, 0, 0, 0, 0, 0, 13});
    CHECK(nn::stage_of(early, kStages) != nn::stage_of(late, kStages));

    network.update(early, 3.0);
    CHECK(network.value(early) != 0.0);
    CHECK(network.value(late) == 0.0);
}

// A staged network must round-trip, and must refuse to load into a network
// with a different stage count — that would silently read the wrong table for
// most boards while still producing finite, plausible numbers.
void test_staged_save_load_round_trip_and_mismatch() {
    constexpr std::size_t kStages = 3;
    const std::vector<nn::TupleSpec> small{nn::TupleSpec{{0, 1, 2}}};
    nn::NTupleNetwork original(small, kStages);
    const auto board = a2048::encode(a2048::CellArray{1, 2, 3, 0, 0, 0, 0, 0,
                                                     0, 0, 0, 0, 0, 0, 0, 12});
    original.update(board, 4.25);

    const auto path = std::filesystem::temp_directory_path() / "a2048_staged_test.bin";
    original.save(path);

    const auto restored = nn::NTupleNetwork::load_from(path);
    CHECK(restored.stage_count() == kStages);
    CHECK(restored.fingerprint() == original.fingerprint());
    CHECK(std::abs(restored.value(board) - original.value(board)) < 1e-9);

    nn::NTupleNetwork wrong_stages(small, 1);
    CHECK_THROWS(wrong_stages.load(path));
    std::filesystem::remove(path);
}

// GATE: the global feature must actually depend on the WHOLE board -- that is
// its entire reason to exist. Two boards identical inside every tuple's cells
// but differing elsewhere must get different global indices.
void test_global_feature_sees_the_whole_board() {
    // Same top-left region (which the default tuples cover), different
    // occupancy in the bottom-right (which they do not).
    a2048::CellArray a{};
    a[0] = 1; a[1] = 2; a[2] = 3; a[3] = 4;
    a2048::CellArray b = a;
    b[15] = 5;  // one extra tile, far from every default tuple's cells

    const auto board_a = a2048::encode(a);
    const auto board_b = a2048::encode(b);
    CHECK(nn::global_feature_index(board_a) != nn::global_feature_index(board_b));

    // And the index must move with BOTH inputs it is built from.
    a2048::CellArray c = a;
    c[3] = 9;  // same empty count, different max exponent
    CHECK(nn::global_feature_index(a2048::encode(c)) != nn::global_feature_index(board_a));
    CHECK(nn::global_feature_index(board_a) < nn::kGlobalFeatureSize);
    CHECK(nn::global_feature_index(board_b) < nn::kGlobalFeatureSize);
}

// Enabling global features must add exactly one active weight per evaluation
// and 256 total, and must leave a network without them bit-identical.
void test_global_features_cost_and_isolation() {
    nn::NTupleNetwork plain(nn::default_tuple_specs());
    nn::NTupleNetwork global(nn::default_tuple_specs(), 1, true);

    CHECK(global.active_weight_count() == plain.active_weight_count() + 1);
    CHECK(global.total_weight_count() == plain.total_weight_count() + nn::kGlobalFeatureSize);
    CHECK(!plain.has_global_features());
    CHECK(global.has_global_features());

    // A network built with global_features=false must be unchanged from one
    // built with the defaulted argument.
    nn::NTupleNetwork defaulted(nn::default_tuple_specs(), 1);
    CHECK(defaulted.fingerprint() == plain.fingerprint());
}

// The global weight must be reachable through active_indices() and move
// value() when updated -- otherwise it is allocated but inert.
void test_global_feature_participates_in_updates() {
    nn::NTupleNetwork network(nn::default_tuple_specs(), 1, true);
    const auto board = a2048::encode(a2048::CellArray{1, 2, 3, 4, 0, 0, 0, 0,
                                                     0, 0, 0, 0, 0, 0, 0, 0});
    std::vector<std::size_t> indices;
    network.active_indices(board, indices);
    CHECK(indices.size() == network.active_weight_count());

    // The last index is the global one; nudging only it must change value().
    const auto before = network.value(board);
    network.weights()[indices.back()] += 2.0F;
    CHECK(std::abs(network.value(board) - before - 2.0) < 1e-6);
}

// Round-trip, and a file written with global features must refuse to load into
// a network without them.
void test_global_features_round_trip_and_mismatch() {
    nn::NTupleNetwork original(nn::default_tuple_specs(), 1, true);
    const auto board = a2048::encode(a2048::CellArray{1, 2, 3, 4, 5, 0, 0, 0,
                                                     0, 0, 0, 0, 0, 0, 0, 0});
    original.update(board, 1.75);

    const auto path = std::filesystem::temp_directory_path() / "a2048_global_test.bin";
    original.save(path);

    const auto restored = nn::NTupleNetwork::load_from(path);
    CHECK(restored.has_global_features());
    CHECK(restored.fingerprint() == original.fingerprint());
    CHECK(std::abs(restored.value(board) - original.value(board)) < 1e-9);

    nn::NTupleNetwork without(nn::default_tuple_specs(), 1, false);
    CHECK_THROWS(without.load(path));
    std::filesystem::remove(path);
}

}  // namespace

int main() {
    const std::vector<std::pair<std::string, void (*)()>> tests{
        {"GATE: index and value match independent reference", test_gate_index_and_value_match_reference},
        {"default network shape and size", test_default_network_shape_and_size},
        {"ordering dedup matches reference", test_ordering_dedup_matches_reference},
        {"fresh network evaluates to zero", test_value_is_zero_on_fresh_network},
        {"update is linear in delta", test_update_is_linear_in_delta},
        {"sparse boards collide and overshoot", test_sparse_boards_collide_and_overshoot},
        {"GATE: rotation invariance holds empirically", test_gate_rotation_invariance_empirically},
        {"extension plane clamps to 15", test_extension_plane_clamps_to_fifteen},
        {"save/load round-trips bit-exactly", test_save_load_round_trips_bit_exactly},
        {"load rejects mismatched and corrupt files", test_load_rejects_mismatched_and_corrupt_files},
        {"invalid specs rejected", test_invalid_specs_are_rejected},
        {"load_from reconstructs shape from file", test_load_from_reconstructs_shape_from_file},
        {"load_from rejects corrupt files", test_load_from_rejects_corrupt_files},
        {"named tuple specs reject unknown names", test_named_tuple_specs_rejects_unknown_names},
        {"named tuple specs have expected sizes", test_named_tuple_specs_have_expected_sizes},
        {"GATE: stage function is monotone over a game", test_stage_function_is_monotone_over_a_game},
        {"single-stage network is unchanged", test_single_stage_is_unchanged},
        {"stages are independent", test_stages_are_independent},
        {"staged save/load round-trips, mismatch rejected",
         test_staged_save_load_round_trip_and_mismatch},
        {"GATE: global feature sees the whole board", test_global_feature_sees_the_whole_board},
        {"global features cost one weight, isolate cleanly",
         test_global_features_cost_and_isolation},
        {"global feature participates in updates",
         test_global_feature_participates_in_updates},
        {"global features round-trip, mismatch rejected",
         test_global_features_round_trip_and_mismatch},
    };

    std::size_t failures = 0;
    for (const auto& [name, test] : tests) {
        try {
            test();
            std::cout << "[PASS] " << name << '\n';
        } catch (const std::exception& error) {
            ++failures;
            std::cerr << "[FAIL] " << name << ": " << error.what() << '\n';
        }
    }

    std::cout << tests.size() - failures << '/' << tests.size() << " tests passed\n";
    return failures == 0 ? 0 : 1;
}
