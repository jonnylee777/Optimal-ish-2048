#include "agents/search_agent.hpp"
#include "core/board.hpp"
#include "evaluation/baseline_heuristic.hpp"
#include "evaluation/evaluator.hpp"
#include "experiments/game_runner.hpp"
#include "search/expectimax.hpp"

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

[[nodiscard]] bool approximately_equal(double left, double right, double tolerance = 1e-9) {
    return std::abs(left - right) <= tolerance;
}

class ZeroEvaluator final : public a2048::Evaluator {
public:
    [[nodiscard]] double evaluate(a2048::Board) const override {
        return 0.0;
    }
};

class TileSumEvaluator final : public a2048::Evaluator {
public:
    [[nodiscard]] double evaluate(a2048::Board board) const override {
        double sum = 0.0;
        for (const auto exponent : a2048::decode(board)) {
            if (exponent != 0) {
                sum += static_cast<double>(std::uint64_t{1} << exponent);
            }
        }
        return sum;
    }
};

[[nodiscard]] double reference_line_monotonicity(
    const std::array<std::uint8_t, a2048::kBoardWidth>& line) {
    std::array<std::uint8_t, a2048::kBoardWidth> occupied{};
    std::size_t count = 0;
    for (const auto exponent : line) {
        if (exponent != 0) {
            occupied[count++] = exponent;
        }
    }

    double increasing = 0.0;
    double decreasing = 0.0;
    for (std::size_t index = 1; index < count; ++index) {
        increasing += std::max(
            static_cast<double>(occupied[index - 1]) - occupied[index], 0.0);
        decreasing += std::max(
            static_cast<double>(occupied[index]) - occupied[index - 1], 0.0);
    }
    return -std::min(increasing, decreasing);
}

[[nodiscard]] a2048::BaselineFeatures reference_features(a2048::Board board) {
    const auto cells = a2048::decode(board);
    a2048::BaselineFeatures features{};
    features.empty_cells = static_cast<double>(a2048::empty_count(board));

    for (std::size_t line = 0; line < a2048::kBoardWidth; ++line) {
        std::array<std::uint8_t, a2048::kBoardWidth> row{};
        std::array<std::uint8_t, a2048::kBoardWidth> column{};
        for (std::size_t offset = 0; offset < a2048::kBoardWidth; ++offset) {
            row[offset] = cells[line * a2048::kBoardWidth + offset];
            column[offset] = cells[offset * a2048::kBoardWidth + line];
        }
        features.monotonicity += reference_line_monotonicity(row);
        features.monotonicity += reference_line_monotonicity(column);
    }

    for (std::size_t row = 0; row < a2048::kBoardWidth; ++row) {
        for (std::size_t column = 0; column < a2048::kBoardWidth; ++column) {
            const auto index = row * a2048::kBoardWidth + column;
            const auto exponent = cells[index];
            if (exponent == 0) {
                continue;
            }
            for (std::size_t next = column + 1; next < a2048::kBoardWidth; ++next) {
                const auto neighbor = cells[row * a2048::kBoardWidth + next];
                if (neighbor != 0) {
                    features.smoothness -= std::abs(
                        static_cast<double>(exponent) - neighbor);
                    break;
                }
            }
            for (std::size_t next = row + 1; next < a2048::kBoardWidth; ++next) {
                const auto neighbor = cells[next * a2048::kBoardWidth + column];
                if (neighbor != 0) {
                    features.smoothness -= std::abs(
                        static_cast<double>(exponent) - neighbor);
                    break;
                }
            }
        }
    }

    const auto maximum = a2048::max_exponent(board);
    if (maximum != 0 &&
        (cells[0] == maximum || cells[3] == maximum ||
         cells[12] == maximum || cells[15] == maximum)) {
        features.corner_preference = maximum;
    }
    return features;
}

void test_precomputed_features_match_reference() {
    std::mt19937_64 generator(0x2048U);
    std::uniform_int_distribution<std::uint64_t> packed_distribution;
    for (std::size_t sample = 0; sample < 25'000; ++sample) {
        const a2048::Board board{packed_distribution(generator), 0};
        const auto actual = a2048::extract_baseline_features(board);
        const auto expected = reference_features(board);
        CHECK(actual.empty_cells == expected.empty_cells);
        CHECK(actual.monotonicity == expected.monotonicity);
        CHECK(actual.smoothness == expected.smoothness);
        CHECK(actual.corner_preference == expected.corner_preference);
    }

    const auto extended = a2048::encode(Cells{
        16, 0, 4, 2,
        17, 5, 0, 1,
        0, 6, 18, 0,
        3, 0, 7, 19,
    });
    const auto actual = a2048::extract_baseline_features(extended);
    const auto expected = reference_features(extended);
    CHECK(actual.empty_cells == expected.empty_cells);
    CHECK(actual.monotonicity == expected.monotonicity);
    CHECK(actual.smoothness == expected.smoothness);
    CHECK(actual.corner_preference == expected.corner_preference);
}

void test_four_baseline_features() {
    const auto ordered = a2048::encode(Cells{1, 2, 3, 4});
    const auto ordered_features = a2048::extract_baseline_features(ordered);
    CHECK(ordered_features.empty_cells == 12.0);
    CHECK(ordered_features.monotonicity == 0.0);
    CHECK(ordered_features.smoothness == -3.0);
    CHECK(ordered_features.corner_preference == 4.0);

    const auto disordered = a2048::encode(Cells{1, 3, 2, 4});
    const auto disordered_features = a2048::extract_baseline_features(disordered);
    CHECK(disordered_features.monotonicity < ordered_features.monotonicity);
    CHECK(disordered_features.smoothness == -5.0);

    const auto maximum_not_in_corner = a2048::encode(Cells{
        1, 0, 0, 0,
        0, 5, 0, 0,
    });
    CHECK(a2048::extract_baseline_features(maximum_not_in_corner).corner_preference == 0.0);
}

void test_weights_are_separate_and_evaluation_is_deterministic() {
    const auto board = a2048::encode(Cells{1, 2, 3, 4});
    const a2048::BaselineHeuristic empty_only({1.0, 0.0, 0.0, 0.0});
    CHECK(empty_only.evaluate(board) == 12.0);

    const a2048::BaselineHeuristic custom({2.0, 3.0, 5.0, 7.0});
    const auto features = a2048::extract_baseline_features(board);
    const auto expected = 2.0 * features.empty_cells +
                          3.0 * features.monotonicity +
                          5.0 * features.smoothness +
                          7.0 * features.corner_preference;
    CHECK(custom.evaluate(board) == expected);
    CHECK(custom.evaluate(board) == custom.evaluate(board));
    CHECK(custom.weights().smoothness == 5.0);
}

void test_depth_one_semantics_and_spawn_probabilities() {
    const auto board = a2048::encode(Cells{1});
    const TileSumEvaluator evaluator;
    a2048::Expectimax search(evaluator);
    const auto result = search.search(board, 1);

    CHECK(result.direction == a2048::Direction::right);
    CHECK(approximately_equal(result.value, 4.2));
    CHECK(result.statistics.player_nodes == 1);
    CHECK(result.statistics.chance_nodes == 2);
    CHECK(result.statistics.spawn_outcomes == 60);
    CHECK(result.statistics.leaf_evaluations == 60);
}

void test_merge_reward_and_terminal_board() {
    const ZeroEvaluator evaluator;
    a2048::Expectimax search(evaluator);

    const auto merge_board = a2048::encode(Cells{1, 1});
    const auto merge_result = search.search(merge_board, 1);
    CHECK(merge_result.direction == a2048::Direction::left);
    CHECK(merge_result.value == 4.0);

    const auto blocked = a2048::encode(Cells{
        1, 2, 1, 2,
        2, 1, 2, 1,
        1, 2, 1, 2,
        2, 1, 2, 1,
    });
    const auto terminal = search.search(blocked, 1);
    CHECK(!terminal.direction.has_value());
    CHECK(terminal.value == 0.0);
    CHECK(terminal.statistics.leaf_evaluations == 1);
}

void test_transposition_cache_matches_uncached_search() {
    const a2048::BaselineHeuristic evaluator;
    const auto board = a2048::encode(Cells{
        1, 2, 3, 0,
        2, 3, 2, 0,
        3, 2, 1, 0,
        1, 1, 2, 0,
    });

    a2048::Expectimax cached(evaluator, {true, 4'096});
    a2048::Expectimax uncached(evaluator, {false, 0});
    const auto cached_result = cached.search(board, 3);
    const auto uncached_result = uncached.search(board, 3);

    CHECK(cached_result.direction == uncached_result.direction);
    CHECK(approximately_equal(cached_result.value, uncached_result.value, 1e-7));
    CHECK(cached_result.statistics.cache_hits > 0);
    CHECK(cached_result.statistics.leaf_evaluations <
          uncached_result.statistics.leaf_evaluations);
}

void test_probability_cutoff_is_explicit_and_cache_safe() {
    const a2048::BaselineHeuristic evaluator;
    const auto board = a2048::encode(Cells{
        1, 2, 3, 0,
        2, 3, 2, 0,
        3, 2, 1, 0,
        1, 1, 2, 0,
    });
    constexpr double cutoff = 0.01;
    a2048::Expectimax cached(evaluator, {true, 4'096, cutoff});
    a2048::Expectimax uncached(evaluator, {false, 0, cutoff});
    const auto cached_result = cached.search(board, 3);
    const auto uncached_result = uncached.search(board, 3);
    CHECK(cached_result.direction == uncached_result.direction);
    CHECK(approximately_equal(cached_result.value, uncached_result.value, 1e-7));

    a2048::Expectimax exact(evaluator);
    const auto exact_result = exact.search(board, 3);
    CHECK(cached_result.statistics.total_nodes() < exact_result.statistics.total_nodes());

    bool rejected = false;
    try {
        a2048::Expectimax invalid(evaluator, {true, 0, 1.0});
    } catch (const std::invalid_argument&) {
        rejected = true;
    }
    CHECK(rejected);
}

void test_symmetry_reduction_preserves_baseline_search_value() {
    const a2048::BaselineHeuristic evaluator;
    const auto board = a2048::encode(Cells{
        1, 2, 3, 0,
        2, 4, 2, 0,
        3, 2, 1, 0,
        1, 1, 2, 0,
    });
    a2048::Expectimax ordinary(evaluator, {true, 16'384});
    a2048::Expectimax symmetric(evaluator, {true, 16'384, 0.0, true});
    const auto ordinary_result = ordinary.search(board, 3);
    const auto symmetric_result = symmetric.search(board, 3);
    CHECK(ordinary_result.direction == symmetric_result.direction);
    CHECK(approximately_equal(ordinary_result.value, symmetric_result.value, 1e-7));

    const auto transposed_result = symmetric.search(a2048::transpose(board), 3);
    CHECK(approximately_equal(symmetric_result.value, transposed_result.value, 1e-7));
}

void test_time_bounded_iterative_deepening_returns_completed_move() {
    const a2048::BaselineHeuristic evaluator;
    const auto board = a2048::encode(Cells{
        1, 2, 3, 0,
        2, 3, 2, 0,
        3, 2, 1, 0,
        1, 1, 2, 0,
    });
    a2048::Expectimax search(evaluator, {true, 65'536, 0.0, true, 0.002});
    const auto result = search.search_iterative(board, 8);
    CHECK(result.direction.has_value());
    CHECK(a2048::move(board, *result.direction).moved);
    CHECK(result.completed_depth < 8);
    CHECK(result.statistics.elapsed_seconds < 1.0);
}

void test_depth_validation_and_search_agent_statistics() {
    const ZeroEvaluator evaluator;
    bool rejected = false;
    try {
        a2048::Expectimax search(evaluator);
        static_cast<void>(search.search(a2048::Board{}, 0));
    } catch (const std::invalid_argument&) {
        rejected = true;
    }
    CHECK(rejected);

    a2048::SearchAgent agent(evaluator, 1, "test-search");
    CHECK(agent.choose_move(a2048::encode(Cells{1, 1})) == a2048::Direction::left);
    CHECK(agent.cumulative_statistics().total_nodes() > 0);
    CHECK(agent.cumulative_statistics().elapsed_seconds > 0.0);
    agent.clear_statistics();
    CHECK(agent.cumulative_statistics().total_nodes() == 0);
}

void test_baseline_agent_runs_complete_games_reproducibly() {
    const a2048::BaselineHeuristic evaluator;
    a2048::SearchAgent first_agent(evaluator, 1, "baseline");
    a2048::SearchAgent second_agent(evaluator, 1, "baseline");
    const a2048::RunConfig config{2, 9'000};
    const auto first = a2048::GameRunner::run(first_agent, config);
    const auto second = a2048::GameRunner::run(second_agent, config);

    CHECK(first.games.size() == 2);
    for (std::size_t index = 0; index < first.games.size(); ++index) {
        CHECK(first.games[index].seed == second.games[index].seed);
        CHECK(first.games[index].score == second.games[index].score);
        CHECK(first.games[index].moves == second.games[index].moves);
        CHECK(first.games[index].max_tile_exponent ==
              second.games[index].max_tile_exponent);
    }
}

}  // namespace

int main() {
    const std::vector<std::pair<std::string, void (*)()>> tests{
        {"four baseline features", test_four_baseline_features},
        {"precomputed feature equivalence", test_precomputed_features_match_reference},
        {"baseline weights", test_weights_are_separate_and_evaluation_is_deterministic},
        {"depth-one semantics", test_depth_one_semantics_and_spawn_probabilities},
        {"merge reward and terminal", test_merge_reward_and_terminal_board},
        {"transposition cache", test_transposition_cache_matches_uncached_search},
        {"probability cutoff", test_probability_cutoff_is_explicit_and_cache_safe},
        {"symmetry reduction", test_symmetry_reduction_preserves_baseline_search_value},
        {"time-bounded iterative deepening", test_time_bounded_iterative_deepening_returns_completed_move},
        {"depth and instrumentation", test_depth_validation_and_search_agent_statistics},
        {"baseline integration", test_baseline_agent_runs_complete_games_reproducibly},
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
