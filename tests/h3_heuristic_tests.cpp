#include "core/board.hpp"
#include "evaluation/h2_heuristic.hpp"
#include "evaluation/h3_heuristic.hpp"
#include "evaluation/structural_heuristic.hpp"

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

// Boards exercised: empty, a plain corner-anchored board, and a stuck-state
// board borrowed in spirit from milestone6_tests.cpp's adverse-stuck
// scenario, so H3's wiring is checked against a board where
// adverse_stuck_penalty is actually nonzero, not just the common zero case.
const std::vector<Cells> kSampleBoards{
    Cells{},
    []() {
        Cells cells{};
        cells[15] = 11;
        cells[14] = 10;
        cells[11] = 9;
        return cells;
    }(),
    // Same adverse-stuck-state board as
    // milestone6_tests.cpp's test_adverse_stuck_state_requires_structural_displacement.
    Cells{
        0, 1, 2, 1,
        0, 2, 1, 2,
        0, 1, 3, 6,
        0, 2, 1, 7,
    },
};

void test_features_match_underlying_h2_and_structural_computation() {
    for (const auto& cells : kSampleBoards) {
        const auto board = a2048::encode(cells);
        const auto h3_features = a2048::extract_h3_features(board, a2048::BoardCorner::bottom_right);
        const auto structural = a2048::extract_structural_features(board, a2048::BoardCorner::bottom_right);
        const auto expected_corner_chain = a2048::corner_chain_score(board);

        CHECK(h3_features.h2.baseline.empty_cells == structural.baseline.empty_cells);
        CHECK(h3_features.h2.baseline.monotonicity == structural.baseline.monotonicity);
        CHECK(h3_features.h2.baseline.smoothness == structural.baseline.smoothness);
        CHECK(h3_features.h2.baseline.corner_preference == structural.baseline.corner_preference);
        CHECK(h3_features.h2.corner_chain == expected_corner_chain);
        CHECK(h3_features.main_line == structural.main_line);
        CHECK(h3_features.structural_stability == structural.structural_stability);
        CHECK(h3_features.adverse_stuck_penalty == structural.adverse_stuck_penalty);
    }
}

void test_weights_are_separate_and_evaluation_is_deterministic() {
    const auto board = a2048::encode(kSampleBoards[1]);
    const a2048::H3Heuristic custom(
        a2048::H3Weights{{{2.0, 3.0, 5.0, 7.0}, 11.0}, 13.0, 17.0, 19.0, 23.0});
    const auto features = a2048::extract_h3_features(board);
    const auto expected = 2.0 * features.h2.baseline.empty_cells +
                          3.0 * features.h2.baseline.monotonicity +
                          5.0 * features.h2.baseline.smoothness +
                          7.0 * features.h2.baseline.corner_preference +
                          11.0 * features.h2.corner_chain +
                          13.0 * features.main_line +
                          17.0 * features.structural_stability -
                          19.0 * features.adverse_stuck_penalty;
    CHECK(custom.evaluate(board) == expected);
    CHECK(custom.evaluate(board) == custom.evaluate(board));
    CHECK(custom.weights().structural_displacement == 23.0);
}

void test_adverse_stuck_penalty_reduces_value_when_present() {
    const auto stuck_board = a2048::encode(kSampleBoards[2]);
    const auto features = a2048::extract_h3_features(stuck_board);
    CHECK(features.adverse_stuck_penalty > 0.0);

    const a2048::H3Heuristic with_penalty(a2048::H3Weights{{}, 0.0, 0.0, 5.0, 0.0});
    const a2048::H3Heuristic without_penalty(a2048::H3Weights{{}, 0.0, 0.0, 0.0, 0.0});
    CHECK(with_penalty.evaluate(stuck_board) < without_penalty.evaluate(stuck_board));
}

void test_evaluate_transition_matches_structural_movement_penalty() {
    const auto old_board = a2048::encode(kSampleBoards[1]);
    const auto move_result = a2048::move(old_board, a2048::Direction::left);
    CHECK(move_result.moved);

    const a2048::H3Heuristic heuristic(a2048::H3Weights{{}, 0.0, 0.0, 0.0, 3.0});
    const auto orientation =
        a2048::extract_structural_features(old_board, a2048::BoardCorner::bottom_right)
            .selected_orientation;
    const auto expected = -3.0 * a2048::structural_movement_penalty(
        old_board, move_result.board, a2048::BoardCorner::bottom_right, orientation);
    CHECK(heuristic.evaluate_transition(old_board, move_result.board) == expected);
}

void test_zero_structural_displacement_weight_short_circuits() {
    const auto old_board = a2048::encode(kSampleBoards[1]);
    const auto move_result = a2048::move(old_board, a2048::Direction::left);
    const a2048::H3Heuristic heuristic(a2048::H3Weights{{}, 0.0, 0.0, 0.0, 0.0});
    CHECK(heuristic.evaluate_transition(old_board, move_result.board) == 0.0);
}

void test_is_not_rotation_invariant() {
    const a2048::H3Heuristic heuristic;
    CHECK(!heuristic.is_rotation_invariant());
}

}  // namespace

int main() {
    const std::vector<std::pair<std::string, void (*)()>> tests{
        {"features match underlying H2 and structural computation", test_features_match_underlying_h2_and_structural_computation},
        {"weights are separate and deterministic", test_weights_are_separate_and_evaluation_is_deterministic},
        {"adverse stuck penalty reduces value", test_adverse_stuck_penalty_reduces_value_when_present},
        {"evaluate_transition matches structural movement penalty", test_evaluate_transition_matches_structural_movement_penalty},
        {"zero structural_displacement weight short-circuits", test_zero_structural_displacement_weight_short_circuits},
        {"is not rotation invariant", test_is_not_rotation_invariant},
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
