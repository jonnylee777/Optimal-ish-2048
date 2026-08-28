#include "agents/greedy_agent.hpp"
#include "agents/random_agent.hpp"
#include "agents/search_agent.hpp"
#include "core/board.hpp"
#include "evaluation/h0_heuristic.hpp"
#include "experiments/game_runner.hpp"
#include "game/game.hpp"

#include <array>
#include <memory>
#include <cstdint>
#include <exception>
#include <iostream>
#include <optional>
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

void test_standard_game_initialization() {
    a2048::Game first(1'234);
    a2048::Game second(1'234);

    CHECK(first.board() == second.board());
    CHECK(a2048::empty_count(first.board()) == 14);
    CHECK(first.score() == 0);
    CHECK(first.move_count() == 0);
    CHECK(first.seed() == 1'234);

    for (const auto exponent : a2048::decode(first.board())) {
        CHECK(exponent == 0 || exponent == 1 || exponent == 2);
    }
}

void test_game_move_scoring_and_spawn() {
    const auto initial = a2048::encode(Cells{1, 1});
    a2048::Game first(initial, 77);
    a2048::Game second(initial, 77);

    const auto first_turn = first.apply_move(a2048::Direction::left);
    const auto second_turn = second.apply_move(a2048::Direction::left);
    CHECK(first_turn.moved);
    CHECK(first_turn.score_gained == 4);
    CHECK(first_turn.spawn.has_value());
    CHECK(first.score() == 4);
    CHECK(first.move_count() == 1);
    CHECK(a2048::empty_count(first.board()) == 14);
    CHECK(first.board() == second.board());
    CHECK(first_turn.spawn->cell_index == second_turn.spawn->cell_index);
    CHECK(first_turn.spawn->exponent == second_turn.spawn->exponent);
}

void test_invalid_move_preserves_all_state() {
    const auto initial = a2048::encode(Cells{1});
    a2048::Game with_invalid_attempt(initial, 91);
    a2048::Game direct_move(initial, 91);

    const auto invalid = with_invalid_attempt.apply_move(a2048::Direction::left);
    CHECK(!invalid.moved);
    CHECK(!invalid.spawn.has_value());
    CHECK(with_invalid_attempt.board() == initial);
    CHECK(with_invalid_attempt.score() == 0);
    CHECK(with_invalid_attempt.move_count() == 0);

    const auto after_invalid = with_invalid_attempt.apply_move(a2048::Direction::right);
    const auto direct = direct_move.apply_move(a2048::Direction::right);
    CHECK(after_invalid.moved);
    CHECK(with_invalid_attempt.board() == direct_move.board());
    CHECK(after_invalid.spawn->cell_index == direct.spawn->cell_index);
    CHECK(after_invalid.spawn->exponent == direct.spawn->exponent);
}

void test_game_over_state() {
    const auto blocked = a2048::encode(Cells{
        1, 2, 1, 2,
        2, 1, 2, 1,
        1, 2, 1, 2,
        2, 1, 2, 1,
    });
    a2048::Game game(blocked, 1);
    CHECK(game.game_over());
    CHECK(!game.apply_move(a2048::Direction::left).moved);
}

void test_random_agent_legality_and_reproducibility() {
    const auto board = a2048::encode(Cells{1});
    a2048::RandomAgent agent;
    std::array<std::optional<a2048::Direction>, 32> first_sequence{};

    agent.reset(555);
    for (auto& direction : first_sequence) {
        direction = agent.choose_move(board);
        CHECK(direction == a2048::Direction::right ||
              direction == a2048::Direction::down);
    }

    agent.reset(555);
    for (const auto expected : first_sequence) {
        CHECK(agent.choose_move(board) == expected);
    }
}

void test_greedy_agent_policy() {
    a2048::GreedyAgent agent;
    agent.reset(999);

    const auto vertical_pair = a2048::encode(Cells{
        1, 0, 0, 0,
        1, 0, 0, 0,
    });
    CHECK(agent.choose_move(vertical_pair) == a2048::Direction::up);

    const auto horizontal_tie = a2048::encode(Cells{1, 1});
    CHECK(agent.choose_move(horizontal_tie) == a2048::Direction::left);

    const auto blocked = a2048::encode(Cells{
        1, 2, 1, 2,
        2, 1, 2, 1,
        1, 2, 1, 2,
        2, 1, 2, 1,
    });
    CHECK(!agent.choose_move(blocked).has_value());
}

void test_experiment_reproducibility() {
    a2048::RandomAgent first_agent;
    a2048::RandomAgent second_agent;
    const a2048::RunConfig config{8, 4'000};
    const auto first = a2048::GameRunner::run(first_agent, config);
    const auto second = a2048::GameRunner::run(second_agent, config);

    CHECK(first.agent_name == "random");
    CHECK(first.games.size() == config.game_count);
    CHECK(first.games.size() == second.games.size());
    for (std::size_t index = 0; index < first.games.size(); ++index) {
        CHECK(first.games[index].seed == config.first_seed + index);
        CHECK(first.games[index].seed == second.games[index].seed);
        CHECK(first.games[index].score == second.games[index].score);
        CHECK(first.games[index].moves == second.games[index].moves);
        CHECK(first.games[index].max_tile_exponent ==
              second.games[index].max_tile_exponent);
    }
    CHECK(first.mean_score() == second.mean_score());
    CHECK(first.best_score() == second.best_score());
    CHECK(first.mean_moves() == second.mean_moves());
    CHECK(first.highest_tile_exponent() == second.highest_tile_exponent());
}

void test_greedy_experiment_and_invalid_config() {
    a2048::GreedyAgent agent;
    const auto result = a2048::GameRunner::run(agent, {5, 8'000});
    CHECK(result.agent_name == "greedy");
    CHECK(result.games.size() == 5);
    CHECK(result.mean_score() > 0.0);
    CHECK(result.best_score() > 0);
    CHECK(result.mean_moves() > 0.0);
    CHECK(result.highest_tile_exponent() > 0);
    CHECK(result.runtime_seconds > 0.0);

    bool rejected = false;
    try {
        static_cast<void>(a2048::GameRunner::run(agent, {0, 1}));
    } catch (const std::invalid_argument&) {
        rejected = true;
    }
    CHECK(rejected);
}

// GATE: the multi-threaded runner must reproduce the serial runner's scores
// exactly. Sample size is the binding constraint on every keep/reject decision
// in this project, and threads are how it is bought -- so "parallel plays the
// same games" is load-bearing, not a nicety.
//
// Uses a SearchAgent rather than a stateless one on purpose: a SearchAgent
// carries a transposition table across games, and the reason assignment cannot
// matter is that entries are stamped with a never-reused generation. A greedy
// agent would pass this test while proving nothing about that.
void test_parallel_runner_matches_serial() {
    const a2048::H0Heuristic evaluator{};
    const a2048::RunConfig config{12, 4'242};

    a2048::SearchAgent serial_agent(evaluator, 2, "expectimax");
    const auto serial = a2048::GameRunner::run(serial_agent, config);
    CHECK(serial.worker_count == 1);

    for (const std::size_t workers : {2U, 3U, 5U}) {
        std::vector<std::unique_ptr<a2048::SearchAgent>> agents;
        std::vector<a2048::Agent*> pointers;
        for (std::size_t index = 0; index < workers; ++index) {
            agents.push_back(
                std::make_unique<a2048::SearchAgent>(evaluator, 2, "expectimax"));
            pointers.push_back(agents.back().get());
        }
        const auto parallel = a2048::GameRunner::run(pointers, config);

        CHECK(parallel.worker_count == std::min(workers, config.game_count));
        CHECK(parallel.games.size() == serial.games.size());
        for (std::size_t index = 0; index < serial.games.size(); ++index) {
            // Seed order, not completion order: results are indexed by game.
            CHECK(parallel.games[index].seed == serial.games[index].seed);
            CHECK(parallel.games[index].score == serial.games[index].score);
            CHECK(parallel.games[index].moves == serial.games[index].moves);
            CHECK(parallel.games[index].max_tile_exponent ==
                  serial.games[index].max_tile_exponent);
        }
        CHECK(parallel.mean_score() == serial.mean_score());

        // Statistics merged across workers must account for every move played,
        // or a parallel run would silently under-report its own search cost.
        std::uint64_t merged_moves = 0;
        for (const auto& agent : agents) {
            const auto& usage = agent->completed_depth_usage();
            for (const auto count : usage) {
                merged_moves += count;
            }
        }
        std::uint64_t played_moves = 0;
        for (const auto& game : parallel.games) {
            played_moves += game.moves;
        }
        CHECK(merged_moves == played_moves);
    }

    // More workers than games must not deadlock, spin, or lose a game.
    std::vector<std::unique_ptr<a2048::SearchAgent>> many;
    std::vector<a2048::Agent*> many_pointers;
    for (std::size_t index = 0; index < 4; ++index) {
        many.push_back(std::make_unique<a2048::SearchAgent>(evaluator, 1, "expectimax"));
        many_pointers.push_back(many.back().get());
    }
    const auto tiny = a2048::GameRunner::run(many_pointers, a2048::RunConfig{2, 99});
    CHECK(tiny.games.size() == 2);
    CHECK(tiny.worker_count == 2);

    bool empty_rejected = false;
    try {
        static_cast<void>(a2048::GameRunner::run(std::vector<a2048::Agent*>{}, config));
    } catch (const std::invalid_argument&) {
        empty_rejected = true;
    }
    CHECK(empty_rejected);
}

}  // namespace

int main() {
    const std::vector<std::pair<std::string, void (*)()>> tests{
        {"standard game initialization", test_standard_game_initialization},
        {"game move scoring and spawn", test_game_move_scoring_and_spawn},
        {"invalid move state preservation", test_invalid_move_preserves_all_state},
        {"game over", test_game_over_state},
        {"random agent", test_random_agent_legality_and_reproducibility},
        {"greedy agent", test_greedy_agent_policy},
        {"experiment reproducibility", test_experiment_reproducibility},
        {"greedy experiment", test_greedy_experiment_and_invalid_config},
        {"GATE: parallel runner reproduces serial scores", test_parallel_runner_matches_serial},
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
