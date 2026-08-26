#include "core/board.hpp"
#include "evaluation/h4_heuristic.hpp"

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

[[nodiscard]] bool approximately_equal(double left, double right, double tolerance = 1e-6) {
    return std::abs(left - right) <= tolerance;
}

// An independent transcription of the reference nneonneo/2048-ai row
// heuristic, written directly from the published source rather than reusing
// our own implementation, so this genuinely cross-checks the port instead of
// comparing our code against itself.
struct ReferenceRow {
    double empty{};
    double merges{};
    double monotonicity{};
    double sum{};
};

[[nodiscard]] ReferenceRow reference_row(unsigned row) {
    const unsigned line[4] = {
        (row >> 0) & 0xf, (row >> 4) & 0xf, (row >> 8) & 0xf, (row >> 12) & 0xf};

    double sum = 0;
    int empty = 0;
    int merges = 0;
    int prev = 0;
    int counter = 0;
    for (int i = 0; i < 4; ++i) {
        const int rank = static_cast<int>(line[i]);
        sum += std::pow(static_cast<double>(rank), 3.5);
        if (rank == 0) {
            empty++;
        } else {
            if (prev == rank) {
                counter++;
            } else if (counter > 0) {
                merges += 1 + counter;
                counter = 0;
            }
            prev = rank;
        }
    }
    if (counter > 0) {
        merges += 1 + counter;
    }

    double monotonicity_left = 0;
    double monotonicity_right = 0;
    for (int i = 1; i < 4; ++i) {
        if (line[i - 1] > line[i]) {
            monotonicity_left += std::pow(static_cast<double>(line[i - 1]), 4.0) -
                                 std::pow(static_cast<double>(line[i]), 4.0);
        } else {
            monotonicity_right += std::pow(static_cast<double>(line[i]), 4.0) -
                                  std::pow(static_cast<double>(line[i - 1]), 4.0);
        }
    }

    return ReferenceRow{static_cast<double>(empty), static_cast<double>(merges),
                        std::min(monotonicity_left, monotonicity_right), sum};
}

// Full-board reference: sum the row heuristic over 4 rows and 4 columns.
[[nodiscard]] double reference_board_score(const Cells& cells, a2048::H4Weights weights) {
    double total = a2048::kH4LostPenaltyTotal;
    for (std::size_t line = 0; line < 4; ++line) {
        unsigned row_bits = 0;
        unsigned column_bits = 0;
        for (std::size_t offset = 0; offset < 4; ++offset) {
            row_bits |= static_cast<unsigned>(cells[line * 4 + offset]) << (offset * 4);
            column_bits |= static_cast<unsigned>(cells[offset * 4 + line]) << (offset * 4);
        }
        for (const auto bits : {row_bits, column_bits}) {
            const auto reference = reference_row(bits);
            total += weights.empty * reference.empty +
                     weights.merges * reference.merges -
                     weights.monotonicity * reference.monotonicity -
                     weights.sum * reference.sum;
        }
    }
    return total;
}

void test_matches_reference_on_random_boards() {
    std::mt19937_64 rng(20260823);
    std::uniform_int_distribution<int> exponent(0, 15);
    const a2048::H4Weights weights{};
    const a2048::H4Heuristic heuristic(weights);

    for (int trial = 0; trial < 5000; ++trial) {
        Cells cells{};
        for (auto& cell : cells) {
            cell = static_cast<std::uint8_t>(exponent(rng));
        }
        const auto board = a2048::encode(cells);
        CHECK(approximately_equal(
            heuristic.evaluate(board), reference_board_score(cells, weights), 1e-3));
    }
}

void test_extended_board_path_matches_reference_shape() {
    // Exponent 16+ forces the exact fallback path instead of the row tables.
    // The reference tables cannot represent exponent 16, so there is no
    // reference value to compare against — assert structural properties.
    Cells ascending{};
    ascending[14] = 15;
    ascending[15] = 16;
    const auto ascending_board = a2048::encode(ascending);
    CHECK(ascending_board.exponent_high_bits != 0);

    const auto features = a2048::extract_h4_features(ascending_board);
    // 8 lines x 4 cells; the two occupied cells each appear in one row and
    // one column, so they remove 4 empty slots in total.
    CHECK(features.empty == 8.0 * 4.0 - 4.0);
    CHECK(features.sum > 0.0);
    // Every occupied line here is monotonically ascending, so the
    // min(left, right) cost is exactly zero.
    CHECK(features.monotonicity == 0.0);

    // A genuinely non-monotonic extended row must cost something positive.
    Cells disordered{};
    disordered[12] = 16;
    disordered[13] = 2;
    disordered[14] = 16;
    const auto disordered_board = a2048::encode(disordered);
    CHECK(disordered_board.exponent_high_bits != 0);
    CHECK(a2048::extract_h4_features(disordered_board).monotonicity > 0.0);
}

void test_empty_board_is_all_empty_cells() {
    const auto board = a2048::encode(Cells{});
    const auto features = a2048::extract_h4_features(board);
    // 4 rows + 4 columns, 4 cells each, all empty.
    CHECK(features.empty == 32.0);
    CHECK(features.merges == 0.0);
    CHECK(features.monotonicity == 0.0);
    CHECK(features.sum == 0.0);
}

void test_merges_counted_for_adjacent_equal_pairs() {
    // One row of four equal tiles: reference counts runs as (1 + run length).
    Cells cells{};
    for (std::size_t index = 0; index < 4; ++index) {
        cells[index] = 3;
    }
    const auto board = a2048::encode(cells);
    const auto features = a2048::extract_h4_features(board);
    const auto expected_row = reference_row(0x3333U);
    // That row contributes its merges once; the four columns each hold a
    // single tile and contribute none.
    CHECK(features.merges == expected_row.merges);
}

void test_monotonic_row_costs_less_than_disordered_row() {
    Cells monotonic{};
    monotonic[0] = 1;
    monotonic[1] = 2;
    monotonic[2] = 3;
    monotonic[3] = 4;

    Cells disordered{};
    disordered[0] = 1;
    disordered[1] = 4;
    disordered[2] = 2;
    disordered[3] = 3;

    const auto monotonic_features = a2048::extract_h4_features(a2048::encode(monotonic));
    const auto disordered_features = a2048::extract_h4_features(a2048::encode(disordered));
    // monotonicity is stored as a positive cost, subtracted at eval time.
    CHECK(monotonic_features.monotonicity < disordered_features.monotonicity);

    const a2048::H4Heuristic heuristic;
    CHECK(heuristic.evaluate(a2048::encode(monotonic)) >
          heuristic.evaluate(a2048::encode(disordered)));
}

void test_weights_are_separate_and_evaluation_is_deterministic() {
    Cells cells{};
    cells[0] = 2;
    cells[1] = 2;
    cells[5] = 4;
    const auto board = a2048::encode(cells);

    const a2048::H4Weights custom_weights{2.0, 3.0, 5.0, 7.0};
    const a2048::H4Heuristic custom(custom_weights);
    const auto features = a2048::extract_h4_features(board);
    const auto expected = a2048::kH4LostPenaltyTotal +
                          2.0 * features.empty +
                          3.0 * features.merges -
                          5.0 * features.monotonicity -
                          7.0 * features.sum;
    CHECK(approximately_equal(custom.evaluate(board), expected));
    CHECK(custom.evaluate(board) == custom.evaluate(board));
    CHECK(custom.weights().merges == 3.0);
}

void test_default_weights_match_published_reference_values() {
    const a2048::H4Weights weights{};
    CHECK(weights.empty == 270.0);
    CHECK(weights.merges == 700.0);
    CHECK(weights.monotonicity == 47.0);
    CHECK(weights.sum == 11.0);
}

void test_is_rotation_invariant_in_practice() {
    const a2048::H4Heuristic heuristic;
    CHECK(heuristic.is_rotation_invariant());

    Cells cells{};
    cells[0] = 5;
    cells[1] = 4;
    cells[2] = 3;
    cells[5] = 2;
    const auto board = a2048::encode(cells);
    // transpose() is one of the eight dihedral symmetries; the claimed
    // invariance must hold for it.
    CHECK(approximately_equal(
        heuristic.evaluate(board), heuristic.evaluate(a2048::transpose(board))));
}

}  // namespace

int main() {
    const std::vector<std::pair<std::string, void (*)()>> tests{
        {"matches reference on 5000 random boards", test_matches_reference_on_random_boards},
        {"extended board path", test_extended_board_path_matches_reference_shape},
        {"empty board", test_empty_board_is_all_empty_cells},
        {"merges counted for adjacent equal pairs", test_merges_counted_for_adjacent_equal_pairs},
        {"monotonic row preferred", test_monotonic_row_costs_less_than_disordered_row},
        {"weights are separate and deterministic", test_weights_are_separate_and_evaluation_is_deterministic},
        {"default weights match published reference", test_default_weights_match_published_reference_values},
        {"rotation invariance holds in practice", test_is_rotation_invariant_in_practice},
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
