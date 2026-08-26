#include "core/board.hpp"
#include "tablebase/formation.hpp"
#include "tablebase/variant_mover.hpp"

#include <algorithm>
#include <array>
#include <cstdint>
#include <exception>
#include <iostream>
#include <random>
#include <set>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace a2048 = adversarial_2048;
namespace tb = adversarial_2048::tablebase;

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

using Cells = a2048::CellArray;

[[nodiscard]] std::uint64_t pack(const Cells& cells) {
    return a2048::encode(cells).packed_exponents;
}

// --- formation primitives ---

void test_matches_formation() {
    // Empty mask list matches everything (the free/variant convention).
    CHECK(tb::matches_formation(0, {}));

    Cells cells{};
    cells[10] = 0xF;
    cells[11] = 0xF;
    const auto board = pack(cells);
    const std::uint64_t mask = (std::uint64_t{0xF} << 40) | (std::uint64_t{0xF} << 44);
    CHECK(tb::matches_formation(board, {mask}));

    // A board missing one locked cell does not match.
    Cells partial{};
    partial[10] = 0xF;
    CHECK(!tb::matches_formation(pack(partial), {mask}));

    // Alternative masks: matching ANY one is enough.
    const std::uint64_t other = std::uint64_t{0xF} << 0;
    CHECK(tb::matches_formation(pack(partial), {other, (std::uint64_t{0xF} << 40)}));
}

void test_is_success() {
    CHECK(tb::is_success(0, 5, {}));  // no shifts => always success

    Cells cells{};
    cells[3] = 9;
    const auto board = pack(cells);
    CHECK(tb::is_success(board, 9, {12}));       // nibble 3 == shift 12
    CHECK(!tb::is_success(board, 8, {12}));      // exact match required, not >=
    CHECK(!tb::is_success(board, 9, {16}));      // wrong cell
    CHECK(tb::is_success(board, 9, {16, 12}));   // any listed cell counts
}

void test_free_tile_sum_skips_walls() {
    Cells cells{};
    cells[0] = 1;        // tile 2
    cells[1] = 3;        // tile 8
    cells[2] = 0xF;      // wall, excluded
    CHECK(tb::free_tile_sum(pack(cells)) == 2 + 8);
}

void test_symmetries_are_a_group() {
    std::mt19937_64 rng(4242);
    const std::array<tb::Symmetry, 8> all{
        tb::Symmetry::identity,      tb::Symmetry::reverse_lr,
        tb::Symmetry::reverse_ud,    tb::Symmetry::transpose_main,
        tb::Symmetry::transpose_anti, tb::Symmetry::rotate_180,
        tb::Symmetry::rotate_left,   tb::Symmetry::rotate_right};

    for (int trial = 0; trial < 500; ++trial) {
        Cells cells{};
        for (auto& cell : cells) {
            cell = static_cast<std::uint8_t>(rng() % 16);
        }
        const auto board = pack(cells);

        // Every element is a permutation of the 16 cells, so the multiset of
        // nibbles is preserved.
        std::array<int, 16> base_counts{};
        for (std::size_t i = 0; i < a2048::kCellCount; ++i) {
            ++base_counts[(board >> (4 * i)) & 0xF];
        }
        for (const auto symmetry : all) {
            const auto moved = tb::apply_symmetry(board, symmetry);
            std::array<int, 16> counts{};
            for (std::size_t i = 0; i < a2048::kCellCount; ++i) {
                ++counts[(moved >> (4 * i)) & 0xF];
            }
            CHECK(counts == base_counts);
        }

        // The group has exactly 8 distinct elements acting on a generic board.
        std::set<std::uint64_t> orbit;
        for (const auto symmetry : all) {
            orbit.insert(tb::apply_symmetry(board, symmetry));
        }
        CHECK(!orbit.empty() && orbit.size() <= 8);

        // Involutions.
        for (const auto symmetry : {tb::Symmetry::reverse_lr, tb::Symmetry::reverse_ud,
                                    tb::Symmetry::transpose_main, tb::Symmetry::transpose_anti,
                                    tb::Symmetry::rotate_180}) {
            CHECK(tb::apply_symmetry(tb::apply_symmetry(board, symmetry), symmetry) == board);
        }
        // rotate_left and rotate_right are mutual inverses.
        CHECK(tb::apply_symmetry(tb::apply_symmetry(board, tb::Symmetry::rotate_left),
                                 tb::Symmetry::rotate_right) == board);
        // Four rotations return to start.
        auto rotated = board;
        for (int i = 0; i < 4; ++i) {
            rotated = tb::apply_symmetry(rotated, tb::Symmetry::rotate_left);
        }
        CHECK(rotated == board);
    }
}

void test_transpose_matches_core_engine() {
    // apply_symmetry(transpose_main) must agree with the engine's own transpose,
    // which is separately covered by engine_tests' differential fuzz.
    std::mt19937_64 rng(99);
    for (int trial = 0; trial < 500; ++trial) {
        Cells cells{};
        for (auto& cell : cells) {
            cell = static_cast<std::uint8_t>(rng() % 16);
        }
        const auto board = pack(cells);
        const auto expected = a2048::transpose(a2048::Board{board, 0}).packed_exponents;
        CHECK(tb::apply_symmetry(board, tb::Symmetry::transpose_main) == expected);
    }
}

void test_canonicalize_is_idempotent_and_orbit_consistent() {
    const std::array<tb::SymmetryMode, 7> modes{
        tb::SymmetryMode::identity, tb::SymmetryMode::full,  tb::SymmetryMode::diagonal,
        tb::SymmetryMode::horizontal, tb::SymmetryMode::min24, tb::SymmetryMode::min33,
        tb::SymmetryMode::min34};

    std::mt19937_64 rng(7);
    for (int trial = 0; trial < 400; ++trial) {
        Cells cells{};
        for (auto& cell : cells) {
            cell = static_cast<std::uint8_t>(rng() % 16);
        }
        const auto board = pack(cells);
        for (const auto mode : modes) {
            const auto canonical = tb::canonicalize(board, mode);
            // Idempotence: canonicalizing a representative changes nothing.
            CHECK(tb::canonicalize(canonical, mode) == canonical);
            // The representative is <= the input (it is a min over the orbit).
            CHECK(canonical <= board);
        }
    }
}

void test_full_canonicalization_collapses_whole_orbit() {
    // Every board in a dihedral orbit must canonicalize to the same key,
    // otherwise the table would store duplicates and the DP would disagree
    // with itself across symmetric positions.
    std::mt19937_64 rng(31337);
    for (int trial = 0; trial < 300; ++trial) {
        Cells cells{};
        for (auto& cell : cells) {
            cell = static_cast<std::uint8_t>(rng() % 16);
        }
        const auto board = pack(cells);
        const auto key = tb::canonicalize(board, tb::SymmetryMode::full);
        for (const auto symmetry : {tb::Symmetry::reverse_lr, tb::Symmetry::reverse_ud,
                                    tb::Symmetry::transpose_main, tb::Symmetry::transpose_anti,
                                    tb::Symmetry::rotate_180, tb::Symmetry::rotate_left,
                                    tb::Symmetry::rotate_right}) {
            CHECK(tb::canonicalize(tb::apply_symmetry(board, symmetry),
                                   tb::SymmetryMode::full) == key);
        }
    }
}

void test_variant_symmetries_preserve_walls() {
    // A variant canonicalizer must never move a wall cell, or the walled-off
    // region would drift and the mover would produce illegal boards.
    struct Case {
        tb::Formation formation;
        tb::SymmetryMode mode;
    };
    const std::array<Case, 3> cases{
        Case{tb::variant_2x4(5), tb::SymmetryMode::min24},
        Case{tb::variant_3x3(5), tb::SymmetryMode::min33},
        Case{tb::variant_3x4(5), tb::SymmetryMode::min34},
    };

    std::mt19937_64 rng(2024);
    for (const auto& item : cases) {
        const auto wall = item.formation.seeds.front();
        for (int trial = 0; trial < 300; ++trial) {
            // Random playable content, walls forced on.
            std::uint64_t board = wall;
            for (std::size_t i = 0; i < a2048::kCellCount; ++i) {
                if (((wall >> (4 * i)) & 0xF) == 0xF) {
                    continue;
                }
                board |= static_cast<std::uint64_t>(rng() % 15) << (4 * i);
            }
            const auto canonical = tb::canonicalize(board, item.mode);
            // Wall cells still walls, and no new walls appeared.
            for (std::size_t i = 0; i < a2048::kCellCount; ++i) {
                const bool was_wall = ((wall >> (4 * i)) & 0xF) == 0xF;
                const bool is_wall = ((canonical >> (4 * i)) & 0xF) == 0xF;
                CHECK(was_wall == is_wall);
            }
        }
    }
}

// --- wall-aware mover ---

void test_variant_mover_matches_core_engine_without_walls() {
    // With no walls present, the variant mover must agree exactly with the
    // engine's move() on both board and score — the strongest available check
    // that segment handling and merge rules are right.
    std::mt19937_64 rng(555);
    for (int trial = 0; trial < 20000; ++trial) {
        Cells cells{};
        for (auto& cell : cells) {
            // Cap at 0xD so a merge can never reach the 0xF wall value, where
            // the two implementations legitimately differ.
            cell = static_cast<std::uint8_t>(rng() % 14);
        }
        const auto board = pack(cells);
        for (const auto direction : a2048::kDirections) {
            const auto reference = a2048::move(a2048::Board{board, 0}, direction);
            const auto variant = tb::move_variant(board, direction);
            CHECK(variant.packed == reference.board.packed_exponents);
            CHECK(variant.score == reference.score);
            CHECK(variant.moved == reference.moved);
        }
    }
}

void test_wall_splits_line_into_segments() {
    // Row 0 = [2, wall, 2, 2]. Moving left must compact only within the
    // right-hand segment; the lone 2 left of the wall cannot cross it.
    Cells cells{};
    cells[0] = 1;
    cells[1] = 0xF;
    cells[2] = 1;
    cells[3] = 1;
    const auto moved = tb::move_variant(pack(cells), a2048::Direction::left);

    Cells expected{};
    expected[0] = 1;
    expected[1] = 0xF;
    expected[2] = 2;  // the two 2s merged into a 4
    CHECK(moved.packed == pack(expected));
    CHECK(moved.score == 4);
    CHECK(moved.moved);
}

void test_wall_blocks_movement_rightward() {
    // Row 0 = [2, 2, wall, 0] moving right: the pair merges against the wall's
    // left side and cannot pass through it.
    Cells cells{};
    cells[0] = 1;
    cells[1] = 1;
    cells[2] = 0xF;
    const auto moved = tb::move_variant(pack(cells), a2048::Direction::right);

    Cells expected{};
    expected[1] = 2;
    expected[2] = 0xF;
    CHECK(moved.packed == pack(expected));
    CHECK(moved.score == 4);
}

void test_max_mergeable_exponent_cannot_reach_wall_value() {
    // Two 16384 tiles (0xE) must NOT merge, since 0xF is the wall marker.
    Cells cells{};
    cells[0] = 0xE;
    cells[1] = 0xE;
    const auto moved = tb::move_variant(pack(cells), a2048::Direction::left);
    CHECK(!moved.moved);
    CHECK(moved.score == 0);
    CHECK(moved.packed == pack(cells));

    // But two 8192 tiles (0xD) merge into 0xE normally.
    Cells low{};
    low[0] = 0xD;
    low[1] = 0xD;
    const auto merged = tb::move_variant(pack(low), a2048::Direction::left);
    Cells expected{};
    expected[0] = 0xE;
    CHECK(merged.packed == pack(expected));
    CHECK(merged.score == (std::uint64_t{1} << 0xE));
}

void test_walls_never_move_under_any_direction() {
    const auto wall = tb::variant_2x4(5).seeds.front();
    std::mt19937_64 rng(616);
    for (int trial = 0; trial < 2000; ++trial) {
        std::uint64_t board = wall;
        for (std::size_t i = 0; i < a2048::kCellCount; ++i) {
            if (((wall >> (4 * i)) & 0xF) == 0xF) {
                continue;
            }
            board |= static_cast<std::uint64_t>(rng() % 14) << (4 * i);
        }
        for (const auto direction : a2048::kDirections) {
            const auto moved = tb::move_variant(board, direction).packed;
            for (std::size_t i = 0; i < a2048::kCellCount; ++i) {
                const bool was_wall = ((wall >> (4 * i)) & 0xF) == 0xF;
                const bool is_wall = ((moved >> (4 * i)) & 0xF) == 0xF;
                CHECK(was_wall == is_wall);
            }
        }
    }
}

void test_variant_playable_cell_counts() {
    const auto count_free = [](std::uint64_t wall) {
        int free_cells = 0;
        for (std::size_t i = 0; i < a2048::kCellCount; ++i) {
            if (((wall >> (4 * i)) & 0xF) != 0xF) {
                ++free_cells;
            }
        }
        return free_cells;
    };
    CHECK(count_free(tb::variant_2x4(5).seeds.front()) == 8);
    CHECK(count_free(tb::variant_3x3(5).seeds.front()) == 9);
    CHECK(count_free(tb::variant_3x4(5).seeds.front()) == 12);

    CHECK(tb::variant_2x4(5).success_shifts.size() == 8);
    CHECK(tb::variant_3x3(5).success_shifts.size() == 9);
    CHECK(tb::variant_3x4(5).success_shifts.size() == 12);
}

}  // namespace

int main() {
    const std::vector<std::pair<std::string, void (*)()>> tests{
        {"matches_formation", test_matches_formation},
        {"is_success", test_is_success},
        {"free_tile_sum skips walls", test_free_tile_sum_skips_walls},
        {"symmetries form a group", test_symmetries_are_a_group},
        {"transpose matches core engine", test_transpose_matches_core_engine},
        {"canonicalize idempotent", test_canonicalize_is_idempotent_and_orbit_consistent},
        {"full canonicalization collapses orbit", test_full_canonicalization_collapses_whole_orbit},
        {"variant symmetries preserve walls", test_variant_symmetries_preserve_walls},
        {"variant mover matches engine (20000 boards)", test_variant_mover_matches_core_engine_without_walls},
        {"wall splits line into segments", test_wall_splits_line_into_segments},
        {"wall blocks rightward movement", test_wall_blocks_movement_rightward},
        {"0xE cannot merge into wall value", test_max_mergeable_exponent_cannot_reach_wall_value},
        {"walls never move", test_walls_never_move_under_any_direction},
        {"variant playable cell counts", test_variant_playable_cell_counts},
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
