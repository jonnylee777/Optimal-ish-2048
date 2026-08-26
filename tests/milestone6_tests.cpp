#include "agents/search_agent.hpp"
#include "core/board.hpp"
#include "evaluation/structural_heuristic.hpp"
#include "optimization/structural_optimizer.hpp"
#include "search/expectimax.hpp"

#include <array>
#include <cmath>
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

[[nodiscard]] a2048::Board board_from_path(
    const a2048::SnakePath& path,
    const std::vector<std::uint8_t>& exponents) {
    Cells cells{};
    for (std::size_t index = 0; index < exponents.size(); ++index) {
        cells[path[index]] = exponents[index];
    }
    return a2048::encode(cells);
}

void test_snake_paths_are_explicit_and_cover_every_cell() {
    const auto horizontal = a2048::get_snake_path(
        a2048::BoardCorner::bottom_right,
        a2048::SnakeOrientation::horizontal_first);
    const a2048::SnakePath expected_horizontal{
        15, 14, 13, 12,
        8, 9, 10, 11,
        7, 6, 5, 4,
        0, 1, 2, 3,
    };
    CHECK(horizontal == expected_horizontal);

    const auto vertical = a2048::get_snake_path(
        a2048::BoardCorner::bottom_right,
        a2048::SnakeOrientation::vertical_first);
    const a2048::SnakePath expected_vertical{
        15, 11, 7, 3,
        2, 6, 10, 14,
        13, 9, 5, 1,
        0, 4, 8, 12,
    };
    CHECK(vertical == expected_vertical);

    for (const auto corner : {
             a2048::BoardCorner::top_left,
             a2048::BoardCorner::top_right,
             a2048::BoardCorner::bottom_left,
             a2048::BoardCorner::bottom_right,
         }) {
        for (const auto orientation : {
                 a2048::SnakeOrientation::horizontal_first,
                 a2048::SnakeOrientation::vertical_first,
             }) {
            std::array<bool, a2048::kCellCount> seen{};
            for (const auto index : a2048::get_snake_path(corner, orientation)) {
                CHECK(index < a2048::kCellCount);
                CHECK(!seen[index]);
                seen[index] = true;
            }
        }
    }
}

void test_required_main_line_ranking() {
    const auto correct = a2048::encode(Cells{
        0, 1, 7, 8,
        0, 1, 6, 9,
        0, 2, 5, 10,
        0, 3, 4, 11,
    });
    const auto disrupted = a2048::encode(Cells{
        0, 1, 4, 8,
        0, 1, 5, 9,
        0, 2, 6, 10,
        0, 3, 7, 11,
    });
    const auto orientation = a2048::SnakeOrientation::vertical_first;
    CHECK(a2048::main_line_score(
              correct, a2048::BoardCorner::bottom_right, orientation) >
          a2048::main_line_score(
              disrupted, a2048::BoardCorner::bottom_right, orientation));
}

void test_anchored_prefix_and_frontier() {
    const auto path = a2048::get_snake_path(
        a2048::BoardCorner::bottom_right,
        a2048::SnakeOrientation::vertical_first);
    const auto board = board_from_path(path, {11, 10, 9, 8, 6, 5});
    CHECK(a2048::anchored_main_line_length(
              board,
              a2048::BoardCorner::bottom_right,
              a2048::SnakeOrientation::vertical_first) == 4);
    const auto frontier = a2048::main_line_frontier(
        board,
        a2048::BoardCorner::bottom_right,
        a2048::SnakeOrientation::vertical_first);
    CHECK(frontier.has_value());
    CHECK(frontier->path_index == 4);
    CHECK(frontier->cell_index == path[4]);
    CHECK(frontier->row == path[4] / a2048::kBoardWidth);
    CHECK(frontier->column == path[4] % a2048::kBoardWidth);
    CHECK(frontier->current_exponent == 6);
    CHECK(frontier->desired_exponent == 7);
}

void test_extended_tiles_are_supported() {
    const auto path = a2048::get_snake_path(
        a2048::BoardCorner::bottom_right,
        a2048::SnakeOrientation::horizontal_first);
    const auto board = board_from_path(path, {17, 16, 15});
    CHECK(a2048::anchored_main_line_length(
              board,
              a2048::BoardCorner::bottom_right,
              a2048::SnakeOrientation::horizontal_first) == 3);
    CHECK(std::isfinite(a2048::main_line_score(
        board,
        a2048::BoardCorner::bottom_right,
        a2048::SnakeOrientation::horizontal_first)));
}

void test_large_structural_movement_costs_more_than_small_movement() {
    const auto path = a2048::get_snake_path(
        a2048::BoardCorner::bottom_right,
        a2048::SnakeOrientation::vertical_first);
    const auto large = board_from_path(path, {11, 10, 9, 8});
    auto moved_large = large;
    moved_large = a2048::with_cell(moved_large, path[0], 0);
    moved_large = a2048::with_cell(moved_large, path[1], 11);

    const auto small = board_from_path(path, {3, 2, 1});
    auto moved_small = small;
    moved_small = a2048::with_cell(moved_small, path[0], 0);
    moved_small = a2048::with_cell(moved_small, path[1], 3);

    const auto large_penalty = a2048::structural_movement_penalty(
        large, moved_large, a2048::BoardCorner::bottom_right,
        a2048::SnakeOrientation::vertical_first);
    const auto small_penalty = a2048::structural_movement_penalty(
        small, moved_small, a2048::BoardCorner::bottom_right,
        a2048::SnakeOrientation::vertical_first);
    CHECK(large_penalty > small_penalty * 10.0);
}

void test_adverse_stuck_state_requires_structural_displacement() {
    const auto board = a2048::encode(Cells{
        0, 1, 2, 1,
        0, 2, 1, 2,
        0, 1, 3, 6,
        0, 2, 1, 7,
    });
    const auto classification = a2048::classify_stuck_state(
        board,
        a2048::BoardCorner::bottom_right,
        a2048::SnakeOrientation::vertical_first);
    CHECK(classification.legal_move_count == 1);
    CHECK(classification.legal_moves[0]);
    CHECK(classification.escape_moves[0]);
    CHECK(classification.merge_move_count == 0);
    CHECK(classification.exposed_prefix_positions[0] > 0);
    CHECK(classification.safe_move_count == 0);
    CHECK(classification.minimum_displacement > 0.0);
    CHECK(classification.is_stuck);
    CHECK(classification.is_adverse);
}

void test_safe_structural_move_is_not_stuck() {
    const auto path = a2048::get_snake_path(
        a2048::BoardCorner::bottom_right,
        a2048::SnakeOrientation::vertical_first);
    const auto board = a2048::with_cell(
        board_from_path(path, {11, 10, 9, 8}), 0, 1);
    const auto classification = a2048::classify_stuck_state(
        board,
        a2048::BoardCorner::bottom_right,
        a2048::SnakeOrientation::vertical_first);
    CHECK(classification.legal_move_count > 0);
    CHECK(classification.safe_move_count > 0);
    CHECK(!classification.is_stuck);
    CHECK(!classification.is_adverse);
}

void test_supporting_tile_improves_structural_stability() {
    const auto supported = a2048::encode(Cells{
        0, 1, 6, 8,
        0, 0, 5, 9,
        0, 0, 4, 10,
        0, 0, 1, 11,
    });
    const auto unsupported = a2048::with_cell(supported, 14, 0);
    const auto orientation = a2048::SnakeOrientation::vertical_first;
    CHECK(a2048::structural_stability_score(
              supported, a2048::BoardCorner::bottom_right, orientation) >
          a2048::structural_stability_score(
              unsupported, a2048::BoardCorner::bottom_right, orientation));
}

class TransitionTestEvaluator final : public a2048::Evaluator {
public:
    [[nodiscard]] double evaluate(a2048::Board) const override {
        return 0.0;
    }

    [[nodiscard]] double evaluate_transition(
        a2048::Board, a2048::Board new_board) const override {
        return a2048::cell_at(new_board, 0) == 2 ? -100.0 : 0.0;
    }
};

void test_search_applies_transition_penalties() {
    const TransitionTestEvaluator evaluator;
    a2048::Expectimax search(evaluator);
    const auto result = search.search(a2048::encode(Cells{1, 1}), 1);
    CHECK(result.direction == a2048::Direction::right);
}

void test_ablation_weights_enable_features_incrementally() {
    const auto main_line = a2048::main_line_ablation_weights();
    CHECK(main_line.main_line > 0.0);
    CHECK(main_line.structural_displacement == 0.0);
    CHECK(main_line.structural_stability == 0.0);
    CHECK(main_line.adverse_stuck == 0.0);

    const auto movement = a2048::movement_ablation_weights();
    CHECK(movement.main_line == main_line.main_line);
    CHECK(movement.structural_displacement > 0.0);
    CHECK(movement.structural_stability == 0.0);
    CHECK(movement.adverse_stuck == 0.0);
}

void test_structural_optimizer_preserves_control_candidates() {
    a2048::StructuralOptimizationConfig config;
    config.candidate_count = 4;
    config.search_depth = 1;
    config.stages = {{1, 1}};
    const auto result = a2048::optimize_structural_weights(config);
    CHECK(result.evaluations.size() == 4);

    const auto& baseline = a2048::kDepth3OptimizedBaselineWeights;
    for (const auto& evaluation : result.evaluations) {
        CHECK(evaluation.weights.baseline.empty_cells == baseline.empty_cells);
        CHECK(evaluation.weights.baseline.monotonicity == baseline.monotonicity);
        CHECK(evaluation.weights.baseline.smoothness == baseline.smoothness);
        CHECK(evaluation.weights.baseline.corner_preference ==
              baseline.corner_preference);
    }
    CHECK(result.evaluations[0].weights.main_line == 0.0);
    CHECK(result.evaluations[0].weights.structural_stability == 0.0);
    CHECK(result.evaluations[0].weights.adverse_stuck == 0.0);
    CHECK(result.evaluations[0].weights.structural_displacement == 0.0);
}

void test_joint_optimizer_varies_baseline_and_structural_weights() {
    a2048::StructuralOptimizationConfig config;
    config.candidate_count = 5;
    config.search_depth = 1;
    config.optimize_baseline_weights = true;
    config.stages = {{1, 1}};
    const auto result = a2048::optimize_structural_weights(config);
    CHECK(result.evaluations.size() == 5);

    const auto& randomized = result.evaluations[4].weights;
    const auto& baseline = a2048::kDepth3OptimizedBaselineWeights;
    CHECK(randomized.baseline.empty_cells != baseline.empty_cells ||
          randomized.baseline.monotonicity != baseline.monotonicity ||
          randomized.baseline.smoothness != baseline.smoothness ||
          randomized.baseline.corner_preference != baseline.corner_preference);
    CHECK(randomized.main_line != 4.0 ||
          randomized.structural_stability != 1.0 ||
          randomized.adverse_stuck != 2.0 ||
          randomized.structural_displacement != 2.0);
    const auto total = [](const a2048::StructuralWeights& weights) {
        return weights.baseline.empty_cells + weights.baseline.monotonicity +
               weights.baseline.smoothness + weights.baseline.corner_preference +
               weights.main_line + weights.structural_stability +
               weights.adverse_stuck + weights.structural_displacement;
    };
    CHECK(std::abs(total(randomized) - total(a2048::StructuralWeights{})) < 1.0e-9);
}

void test_adaptive_depth_boundaries() {
    const a2048::AdaptiveDepthSchedule schedule{4, 6, 8};
    const auto board_with_occupied = [](std::size_t occupied) {
        Cells cells{};
        for (std::size_t index = 0; index < occupied; ++index) {
            cells[index] = 1;
        }
        return a2048::encode(cells);
    };
    CHECK(schedule.select(board_with_occupied(0)) == 4);
    CHECK(schedule.select(board_with_occupied(6)) == 4);
    CHECK(schedule.select(board_with_occupied(7)) == 6);
    CHECK(schedule.select(board_with_occupied(10)) == 6);
    CHECK(schedule.select(board_with_occupied(11)) == 8);
    CHECK(schedule.select(board_with_occupied(16)) == 8);
    CHECK(schedule.maximum_depth() == 8);
}

void test_v2_1_frozen_weights() {
    const auto& weights = a2048::kDepth4OptimizedStructuralWeights;
    CHECK(weights.baseline.empty_cells == 115.82974924638357);
    CHECK(weights.baseline.monotonicity == 37.30236719801037);
    CHECK(weights.baseline.smoothness == 24.712293838159923);
    CHECK(weights.baseline.corner_preference == 27.468659832176016);
    CHECK(weights.main_line == 1.3726429758766383);
    CHECK(weights.structural_stability == 1.17441510554765);
    CHECK(weights.adverse_stuck == 3.0596792676181983);
    CHECK(weights.structural_displacement == 0.5608988108966877);
}

}  // namespace

int main() {
    const std::vector<std::pair<std::string, void (*)()>> tests{
        {"snake paths", test_snake_paths_are_explicit_and_cover_every_cell},
        {"main-line ranking", test_required_main_line_ranking},
        {"anchored prefix and frontier", test_anchored_prefix_and_frontier},
        {"extended tiles", test_extended_tiles_are_supported},
        {"large-tile movement priority", test_large_structural_movement_costs_more_than_small_movement},
        {"adverse stuck state", test_adverse_stuck_state_requires_structural_displacement},
        {"safe structural move", test_safe_structural_move_is_not_stuck},
        {"structural stability", test_supporting_tile_improves_structural_stability},
        {"search transition penalties", test_search_applies_transition_penalties},
        {"ablation configuration", test_ablation_weights_enable_features_incrementally},
        {"optimizer control candidates", test_structural_optimizer_preserves_control_candidates},
        {"joint optimizer candidates", test_joint_optimizer_varies_baseline_and_structural_weights},
        {"adaptive depth boundaries", test_adaptive_depth_boundaries},
        {"V2.1 frozen weights", test_v2_1_frozen_weights},
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
