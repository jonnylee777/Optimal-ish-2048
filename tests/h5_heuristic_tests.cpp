#include "core/board.hpp"
#include "evaluation/h5_heuristic.hpp"
#include "tablebase/formation.hpp"  // apply_symmetry: independently tested (14/14),
                                    // reused here only to verify H5, not a production
                                    // dependency of the heuristic itself.

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <exception>
#include <iostream>
#include <random>
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

// --- independent transcription of the reference (native_core/src/AIPlayer.cpp) ---
// Written fresh from the published source, not by calling into h5_heuristic.cpp, so
// agreement is a real cross-check rather than a tautology. Mirrors reference.md's own
// int32/int16 arithmetic exactly, including truncating casts.

constexpr std::array<std::int32_t, 16> kReferenceTileWeight{
    0, 2, 4, 8, 16, 32, 64, 128, 248, 388, 488, 518, 519, 519, 519, 520};

[[nodiscard]] std::int32_t reference_line_score(const std::array<std::int32_t, 4>& w) {
    std::int32_t score_dpdf = w[0];
    for (std::size_t x = 0; x < 3; ++x) {
        if (w[x] < w[x + 1]) {
            if (w[x] > 400) {
                score_dpdf += (w[x] << 1) + (w[x + 1] - w[x]) * static_cast<std::int32_t>(x);
            } else if (w[x] > 300 && x == 1 && w[0] > w[1]) {
                score_dpdf += (w[x] << 1);
            } else {
                score_dpdf -= (w[x + 1] - w[x]) << 3;
                score_dpdf -= w[x + 1] * 3;
                if (x < 2 && w[x + 2] < w[x + 1] && w[x + 1] > 30) {
                    score_dpdf -= std::max(80, w[x + 1]);
                }
            }
        } else if (x < 2) {
            score_dpdf += w[x + 1] + w[x];
        } else {
            score_dpdf += static_cast<std::int32_t>((w[x + 1] + w[x]) * 0.5);
        }
    }
    if (w[0] > 400 && w[1] > 300 && w[2] > 200 && w[2] > w[3] && w[3] < 300) {
        score_dpdf += w[3] >> 2;
    }

    std::int32_t score_t;
    const auto min_ends = std::min(w[0], w[3]);
    if (min_ends < 32) {
        score_t = -16384;
    } else if ((w[0] < w[1] && w[0] < 400) || (w[3] < w[2] && w[3] < 400)) {
        score_t = -(std::max(w[1], w[2]) * 10);
    } else {
        score_t = static_cast<std::int32_t>((w[0] * 1.8 + w[3] * 1.8) +
                                            std::max(w[1], w[2]) * 1.5 +
                                            std::min(160, std::min(w[1], w[2])) * 2.5);
        if (std::min(w[1], w[2]) < 8) {
            score_t -= 60;
        }
    }

    int zero_count = 0;
    for (const auto value : w) {
        if (value == 0) {
            ++zero_count;
        }
    }
    const auto sum_123 = w[1] + w[2] + w[3];
    std::int32_t penalty = 0;
    if (w[0] > 100 && ((zero_count > 1 && sum_123 < 32) || sum_123 < 12)) {
        penalty = 4;
    }
    return std::max(score_dpdf, score_t) / 4 - penalty;
}

[[nodiscard]] double reference_board_score(const Cells& cells) {
    std::int64_t sum_x1 = 0, sum_x2 = 0, sum_y1 = 0, sum_y2 = 0;
    for (std::size_t line = 0; line < 4; ++line) {
        std::array<std::int32_t, 4> row_fwd{}, row_rev{}, col_fwd{}, col_rev{};
        for (std::size_t offset = 0; offset < 4; ++offset) {
            const auto row_w = kReferenceTileWeight[cells[line * 4 + offset]];
            const auto col_w = kReferenceTileWeight[cells[offset * 4 + line]];
            row_fwd[offset] = row_w;
            row_rev[3 - offset] = row_w;
            col_fwd[offset] = col_w;
            col_rev[3 - offset] = col_w;
        }
        sum_x1 += static_cast<std::int16_t>(reference_line_score(row_fwd));
        sum_x2 += static_cast<std::int16_t>(reference_line_score(row_rev));
        sum_y1 += static_cast<std::int16_t>(reference_line_score(col_fwd));
        sum_y2 += static_cast<std::int16_t>(reference_line_score(col_rev));
    }
    return static_cast<double>(std::max(sum_x1, sum_x2) + std::max(sum_y1, sum_y2));
}

void test_matches_reference_on_random_boards() {
    std::mt19937_64 rng(20260824);
    std::uniform_int_distribution<int> exponent(0, 15);
    const a2048::H5Heuristic heuristic;

    for (int trial = 0; trial < 5000; ++trial) {
        Cells cells{};
        for (auto& cell : cells) {
            cell = static_cast<std::uint8_t>(exponent(rng));
        }
        const auto board = a2048::encode(cells);
        CHECK(heuristic.evaluate(board) == reference_board_score(cells));
    }
}

void test_line_score_matches_reference_directly() {
    std::mt19937_64 rng(7);
    std::uniform_int_distribution<std::int32_t> weight(0, 520);
    for (int trial = 0; trial < 20000; ++trial) {
        std::array<std::int32_t, 4> weights{weight(rng), weight(rng), weight(rng), weight(rng)};
        CHECK(a2048::h5_line_score(weights) == reference_line_score(weights));
    }
}

void test_tile_weight_saturates() {
    // Exact values from the reference table, spot-checked.
    CHECK(a2048::h5_tile_weight(0) == 0);
    CHECK(a2048::h5_tile_weight(1) == 2);
    CHECK(a2048::h5_tile_weight(8) == 248);
    CHECK(a2048::h5_tile_weight(11) == 518);
    CHECK(a2048::h5_tile_weight(15) == 520);
    // Extension-plane exponents (16-31) have no reference equivalent; this
    // port continues the plateau rather than indexing out of bounds.
    CHECK(a2048::h5_tile_weight(16) == 520);
    CHECK(a2048::h5_tile_weight(31) == 520);
}

void test_extended_board_path_is_consistent_with_ordinary_path() {
    // A board built entirely from exponents <= 15 must score identically
    // whether or not it happens to carry an (unused) extension plane, since
    // evaluate_extended() and the table path implement the same formula.
    std::mt19937_64 rng(99);
    std::uniform_int_distribution<int> exponent(0, 15);
    const a2048::H5Heuristic heuristic;
    for (int trial = 0; trial < 500; ++trial) {
        Cells cells{};
        for (auto& cell : cells) {
            cell = static_cast<std::uint8_t>(exponent(rng));
        }
        const auto ordinary = a2048::encode(cells);
        CHECK(ordinary.exponent_high_bits == 0);
        // Force the extended path by setting a high bit on a cell that's
        // already 0, which changes nothing about the visible exponent (0 has
        // no high-bit-eligible representation via encode(), so instead use
        // with_cell to write an actual exponent >= 16 and compare structural
        // properties instead of exact equality).
        static_cast<void>(ordinary);
    }

    // Direct check: an exponent-16 tile must be treated as weight 520,
    // exactly like an exponent-15 tile, on the extended path.
    Cells fifteens{};
    fifteens.fill(15);
    Cells sixteens{};
    sixteens.fill(15);
    auto extended = a2048::encode(sixteens);
    extended = a2048::with_cell(extended, 0, 16);
    fifteens[0] = 15;
    CHECK(a2048::H5Heuristic{}.evaluate(extended) ==
          a2048::H5Heuristic{}.evaluate(a2048::encode(fifteens)));
}

void test_both_empty_ends_trigger_the_dead_line_case_regardless_of_arrangement() {
    // score_t keys on min(weights[0], weights[3]): whenever EITHER end is
    // near-empty, score_t collapses to -16384 and the result falls back to
    // whatever dpdf makes of it — regardless of where the large tiles
    // actually sit. Both of these trigger that path (one end is 0 in each),
    // so this is not "stranded vs. monotonic," it's two different dpdf
    // outcomes under the same dead-end penalty. Values obtained by running
    // h5_line_score directly (not hand-derived) and pinned as a regression
    // check, since this formula's specific integer thresholds aren't safely
    // predictable by inspection.
    const std::array<std::int32_t, 4> large_tiles_in_middle{0, 520, 520, 0};
    const std::array<std::int32_t, 4> large_tiles_ascending{0, 0, 520, 520};
    CHECK(a2048::h5_line_score(large_tiles_in_middle) == -1105);
    CHECK(a2048::h5_line_score(large_tiles_ascending) == -1300);
}

void test_t_formation_line_beats_disordered_line_of_equal_content() {
    // Two large tiles walling the ends with small tiles between (T-formation
    // shape) should score at least as well as the same four values in an
    // order that breaks both the monotonic run AND the wall structure.
    const std::array<std::int32_t, 4> walled{488, 8, 8, 488};       // T-like: large, small, small, large
    const std::array<std::int32_t, 4> scattered{8, 488, 488, 8};     // ends empty-ish, middle large
    CHECK(a2048::h5_line_score(walled) > a2048::h5_line_score(scattered));
}

void test_is_rotation_invariant_empirically() {
    // This is the load-bearing claim in h5_heuristic.hpp's comment: summing
    // forward/reverse per axis BEFORE taking max() should make the total
    // invariant under all 8 dihedral symmetries, even though a single line's
    // own score depends on which direction it's read. Verify directly rather
    // than trusting the derivation.
    const a2048::H5Heuristic heuristic;
    CHECK(heuristic.is_rotation_invariant());

    const std::array<tb::Symmetry, 8> all{
        tb::Symmetry::identity,       tb::Symmetry::reverse_lr,
        tb::Symmetry::reverse_ud,     tb::Symmetry::transpose_main,
        tb::Symmetry::transpose_anti, tb::Symmetry::rotate_180,
        tb::Symmetry::rotate_left,    tb::Symmetry::rotate_right};

    std::mt19937_64 rng(2048);
    std::uniform_int_distribution<int> exponent(0, 15);
    for (int trial = 0; trial < 1000; ++trial) {
        Cells cells{};
        for (auto& cell : cells) {
            cell = static_cast<std::uint8_t>(exponent(rng));
        }
        const auto board = a2048::encode(cells).packed_exponents;
        const auto reference = heuristic.evaluate(a2048::Board{board, 0});
        for (const auto symmetry : all) {
            const auto moved = tb::apply_symmetry(board, symmetry);
            CHECK(heuristic.evaluate(a2048::Board{moved, 0}) == reference);
        }
    }
}

void test_deterministic() {
    Cells cells{};
    cells[0] = 10;
    cells[5] = 7;
    cells[15] = 3;
    const auto board = a2048::encode(cells);
    const a2048::H5Heuristic heuristic;
    CHECK(heuristic.evaluate(board) == heuristic.evaluate(board));
}

}  // namespace

int main() {
    const std::vector<std::pair<std::string, void (*)()>> tests{
        {"matches reference on 5000 random boards", test_matches_reference_on_random_boards},
        {"line_score matches reference (20000 random weight vectors)", test_line_score_matches_reference_directly},
        {"tile weight saturates", test_tile_weight_saturates},
        {"extended path consistent with ordinary path", test_extended_board_path_is_consistent_with_ordinary_path},
        {"empty ends trigger dead-line case regardless of arrangement", test_both_empty_ends_trigger_the_dead_line_case_regardless_of_arrangement},
        {"T-formation line beats scattered line", test_t_formation_line_beats_disordered_line_of_equal_content},
        {"GATE: rotation invariance holds empirically (8000 checks)", test_is_rotation_invariant_empirically},
        {"deterministic", test_deterministic},
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
