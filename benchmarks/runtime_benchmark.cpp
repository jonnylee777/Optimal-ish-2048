#include "core/board.hpp"
#include "agents/search_agent.hpp"
#include "evaluation/baseline_heuristic.hpp"
#include "evaluation/structural_heuristic.hpp"
#include "game/game.hpp"
#include "search/expectimax.hpp"

#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <tuple>

namespace a2048 = adversarial_2048;

namespace {

[[nodiscard]] std::uint64_t next_random(std::uint64_t& state) noexcept {
    state ^= state << 13U;
    state ^= state >> 7U;
    state ^= state << 17U;
    return state;
}

}  // namespace

int main() {
    constexpr std::size_t board_count = 4'096;
    constexpr std::size_t evaluation_count = 5'000'000;
    std::array<a2048::Board, board_count> boards{};
    std::uint64_t random_state = 0x5EED2048ULL;
    for (auto& board : boards) {
        a2048::CellArray cells{};
        for (auto& exponent : cells) {
            const auto sample = next_random(random_state);
            exponent = sample % 4U == 0U
                ? 0U
                : static_cast<std::uint8_t>(1U + sample % 10U);
        }
        board = a2048::encode(cells);
    }

    const a2048::BaselineHeuristic evaluator(a2048::kDepth3OptimizedBaselineWeights);
    double checksum = 0.0;
    const auto evaluator_start = std::chrono::steady_clock::now();
    for (std::size_t index = 0; index < evaluation_count; ++index) {
        checksum += evaluator.evaluate(boards[index & (board_count - 1U)]);
    }
    const auto evaluator_seconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - evaluator_start).count();

    constexpr std::size_t structural_evaluation_count = 250'000;
    const a2048::StructuralHeuristic main_line_evaluator(
        a2048::main_line_ablation_weights());
    const a2048::StructuralHeuristic movement_evaluator(
        a2048::movement_ablation_weights());
    const a2048::StructuralHeuristic full_structural_evaluator;

    a2048::Game game(10'026);
    a2048::SearchAgent setup_agent(evaluator, 1, "benchmark-setup");
    for (std::size_t move_index = 0; move_index < 300 && !game.game_over(); ++move_index) {
        const auto direction = setup_agent.choose_move(game.board());
        if (!direction.has_value() || !game.apply_move(*direction).moved) {
            break;
        }
    }
    const auto search_board = game.board();
    std::cout << std::fixed << std::setprecision(2)
              << "evaluator calls:     " << evaluation_count << '\n'
              << "evaluator calls/sec: "
              << static_cast<double>(evaluation_count) / evaluator_seconds << '\n'
              << "evaluator checksum:  " << checksum << '\n';

    for (const auto& [name, structural_evaluator] :
         std::array<std::pair<const char*, const a2048::Evaluator*>, 3>{
             std::pair{"main-line", &main_line_evaluator},
             std::pair{"movement", &movement_evaluator},
             std::pair{"full-structural", &full_structural_evaluator},
         }) {
        const auto start = std::chrono::steady_clock::now();
        for (std::size_t index = 0; index < structural_evaluation_count; ++index) {
            checksum += structural_evaluator->evaluate(
                boards[index & (board_count - 1U)]);
        }
        const auto seconds = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - start).count();
        std::cout << name << " evaluator calls/sec: "
                  << static_cast<double>(structural_evaluation_count) / seconds << '\n';
    }

    for (const std::uint32_t depth : {3U, 4U, 5U}) {
        a2048::Expectimax search(evaluator);
        const auto result = search.search(search_board, depth);
        std::cout << std::setprecision(6)
                  << "depth " << depth << " seconds:      "
                  << result.statistics.elapsed_seconds << '\n'
                  << "depth " << depth << " nodes:        "
                  << result.statistics.total_nodes() << '\n'
                  << std::setprecision(2)
                  << "depth " << depth << " nodes/sec:    "
                  << result.statistics.nodes_per_second() << '\n'
                  << "depth " << depth << " value:        " << result.value << '\n';
        checksum += result.value;
    }

    for (const auto& [name, options] :
         std::array<std::pair<const char*, a2048::ExpectimaxOptions>, 2>{
             std::pair{
                 "large-cache",
                 a2048::ExpectimaxOptions{true, 1U << 20U, 0.0, false, 0.0}},
             std::pair{
                 "large-cache+symmetry",
                 a2048::ExpectimaxOptions{true, 1U << 20U, 0.0, true, 0.0}},
         }) {
        a2048::Expectimax search(evaluator, options);
        const auto result = search.search(search_board, 5);
        std::cout << name << " depth 5 seconds: " << std::setprecision(6)
                  << result.statistics.elapsed_seconds << '\n'
                  << name << " depth 5 nodes:   "
                  << result.statistics.total_nodes() << '\n'
                  << name << " cache hit rate:  " << std::setprecision(2)
                  << result.statistics.cache_hit_rate() * 100.0 << "%\n";
        checksum += result.value;
    }

    for (const auto& [name, structural_evaluator] :
         std::array<std::pair<const char*, const a2048::Evaluator*>, 3>{
             std::pair{"main-line", &main_line_evaluator},
             std::pair{"movement", &movement_evaluator},
             std::pair{"full-structural", &full_structural_evaluator},
         }) {
        a2048::Expectimax search(*structural_evaluator);
        const auto result = search.search(search_board, 3);
        std::cout << name << " depth 3 seconds: " << std::setprecision(6)
                  << result.statistics.elapsed_seconds << '\n'
                  << name << " depth 3 nodes/sec: " << std::setprecision(2)
                  << result.statistics.nodes_per_second() << '\n'
                  << name << " depth 3 value: " << result.value << '\n';
        checksum += result.value;
    }

    constexpr double cutoff = 0.0001;
    for (const std::uint32_t depth : {4U, 5U}) {
        a2048::Expectimax search(evaluator, {true, 1U << 16U, cutoff});
        const auto result = search.search(search_board, depth);
        std::cout << std::setprecision(6)
                  << "pruned depth " << depth << " seconds: "
                  << result.statistics.elapsed_seconds << '\n'
                  << "pruned depth " << depth << " nodes:   "
                  << result.statistics.total_nodes() << '\n'
                  << std::setprecision(2)
                  << "pruned depth " << depth << " value:   "
                  << result.value << '\n';
        checksum += result.value;
    }

    for (const auto& [name, bounded_evaluator, capacity, symmetry] :
         std::array<
             std::tuple<const char*, const a2048::Evaluator*, std::size_t, bool>,
             6>{
             std::tuple{"baseline-64k", &evaluator, 1U << 16U, false},
             std::tuple{"baseline-1m", &evaluator, 1U << 20U, false},
             std::tuple{"baseline-64k-symmetry", &evaluator, 1U << 16U, true},
             std::tuple{"baseline-1m-symmetry", &evaluator, 1U << 20U, true},
             std::tuple{"full-structural-64k", &full_structural_evaluator, 1U << 16U, false},
             std::tuple{"full-structural-1m", &full_structural_evaluator, 1U << 20U, false},
         }) {
        a2048::Expectimax search(
            *bounded_evaluator,
            {true, capacity, cutoff, symmetry, 0.05});
        const auto result = search.search_iterative(search_board, 8);
        std::cout << "bounded " << name << " max-depth 8 seconds: "
                  << std::setprecision(6) << result.statistics.elapsed_seconds << '\n'
                  << "bounded " << name << " completed depth: "
                  << result.completed_depth << '\n'
                  << "bounded " << name << " nodes: "
                  << result.statistics.total_nodes() << '\n';
        checksum += result.value;
    }
    return checksum == 0.0 ? 1 : 0;
}
