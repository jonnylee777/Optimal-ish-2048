#include "core/board.hpp"
#include "evaluation/h2_heuristic.hpp"

#include <cstdint>
#include <exception>
#include <iostream>
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

void test_empty_board_has_no_corner_chain() {
    const auto board = a2048::encode(Cells{});
    const auto features = a2048::extract_h2_features(board);
    CHECK(features.baseline.empty_cells == 16.0);
    CHECK(features.corner_chain == 0.0);
}

void test_corner_max_with_no_matching_neighbor_scores_zero() {
    // Index 15 = bottom-right corner. Neighbors (14, 11) are empty.
    Cells cells{};
    cells[15] = 8;
    const auto board = a2048::encode(cells);
    CHECK(a2048::extract_h2_features(board).corner_chain == 0.0);
}

void test_corner_max_with_one_matching_neighbor() {
    // Index 15 = bottom-right corner (exponent 8). Index 14 (left neighbor)
    // holds exponent 7, exactly one less.
    Cells cells{};
    cells[15] = 8;
    cells[14] = 7;
    const auto board = a2048::encode(cells);
    CHECK(a2048::extract_h2_features(board).corner_chain == 7.0);
}

void test_corner_max_with_both_matching_neighbors() {
    // Index 0 = top-left corner (exponent 10). Neighbors are index 1 (right)
    // and index 4 (down), both exponent 9.
    Cells cells{};
    cells[0] = 10;
    cells[1] = 9;
    cells[4] = 9;
    const auto board = a2048::encode(cells);
    CHECK(a2048::extract_h2_features(board).corner_chain == 18.0);
}

void test_non_corner_max_scores_zero_regardless_of_neighbors() {
    // Index 5 (row 1, col 1) is not a corner.
    Cells cells{};
    cells[5] = 8;
    cells[1] = 7;
    cells[4] = 7;
    const auto board = a2048::encode(cells);
    CHECK(a2048::extract_h2_features(board).corner_chain == 0.0);
}

void test_wrong_gap_neighbor_does_not_count() {
    // Neighbor holds exponent 6, not the required maximum - 1 = 7.
    Cells cells{};
    cells[15] = 8;
    cells[14] = 6;
    const auto board = a2048::encode(cells);
    CHECK(a2048::extract_h2_features(board).corner_chain == 0.0);
}

void test_zero_defaults_to_h1_behavior() {
    Cells cells{};
    cells[15] = 8;
    cells[14] = 7;
    const auto board = a2048::encode(cells);
    const a2048::H2Heuristic no_chain({{}, 0.0});
    const auto baseline_only = a2048::extract_baseline_features(board);
    const a2048::BaselineWeights default_weights{};
    const auto expected = default_weights.empty_cells * baseline_only.empty_cells +
                          default_weights.monotonicity * baseline_only.monotonicity +
                          default_weights.smoothness * baseline_only.smoothness +
                          default_weights.corner_preference * baseline_only.corner_preference;
    CHECK(no_chain.evaluate(board) == expected);
}

void test_weights_are_separate_and_evaluation_is_deterministic() {
    Cells cells{};
    cells[15] = 8;
    cells[14] = 7;
    const auto board = a2048::encode(cells);

    const a2048::H2Heuristic custom({{2.0, 3.0, 5.0, 7.0}, 11.0});
    const auto features = a2048::extract_h2_features(board);
    const auto expected = 2.0 * features.baseline.empty_cells +
                          3.0 * features.baseline.monotonicity +
                          5.0 * features.baseline.smoothness +
                          7.0 * features.baseline.corner_preference +
                          11.0 * features.corner_chain;
    CHECK(custom.evaluate(board) == expected);
    CHECK(custom.evaluate(board) == custom.evaluate(board));
    CHECK(custom.weights().corner_chain == 11.0);
}

void test_is_rotation_invariant() {
    const a2048::H2Heuristic heuristic;
    CHECK(heuristic.is_rotation_invariant());
}

}  // namespace

int main() {
    const std::vector<std::pair<std::string, void (*)()>> tests{
        {"empty board has no corner chain", test_empty_board_has_no_corner_chain},
        {"corner max with no matching neighbor scores zero", test_corner_max_with_no_matching_neighbor_scores_zero},
        {"corner max with one matching neighbor", test_corner_max_with_one_matching_neighbor},
        {"corner max with both matching neighbors", test_corner_max_with_both_matching_neighbors},
        {"non-corner max scores zero", test_non_corner_max_scores_zero_regardless_of_neighbors},
        {"wrong gap neighbor does not count", test_wrong_gap_neighbor_does_not_count},
        {"zero corner_chain weight matches H1", test_zero_defaults_to_h1_behavior},
        {"weights are separate and deterministic", test_weights_are_separate_and_evaluation_is_deterministic},
        {"is rotation invariant", test_is_rotation_invariant},
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
