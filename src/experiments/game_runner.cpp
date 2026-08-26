#include "experiments/game_runner.hpp"

#include "core/board.hpp"
#include "game/game.hpp"

#include <chrono>
#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

namespace adversarial_2048 {
namespace {

[[nodiscard]] std::uint64_t decision_seed(std::uint64_t game_seed) noexcept {
    // SplitMix64 finalizer: deterministic separation between environment and
    // agent randomness without coupling either RNG stream to execution order.
    auto value = game_seed + 0x9E3779B97F4A7C15ULL;
    value = (value ^ (value >> 30U)) * 0xBF58476D1CE4E5B9ULL;
    value = (value ^ (value >> 27U)) * 0x94D049BB133111EBULL;
    return value ^ (value >> 31U);
}

}  // namespace

double ExperimentResult::mean_score() const noexcept {
    if (games.empty()) {
        return 0.0;
    }
    long double total = 0;
    for (const auto& game : games) {
        total += static_cast<long double>(game.score);
    }
    return static_cast<double>(total / static_cast<long double>(games.size()));
}

std::uint64_t ExperimentResult::best_score() const noexcept {
    std::uint64_t best = 0;
    for (const auto& game : games) {
        if (game.score > best) {
            best = game.score;
        }
    }
    return best;
}

double ExperimentResult::mean_moves() const noexcept {
    if (games.empty()) {
        return 0.0;
    }
    long double total = 0;
    for (const auto& game : games) {
        total += static_cast<long double>(game.moves);
    }
    return static_cast<double>(total / static_cast<long double>(games.size()));
}

std::uint8_t ExperimentResult::highest_tile_exponent() const noexcept {
    std::uint8_t highest = 0;
    for (const auto& game : games) {
        if (game.max_tile_exponent > highest) {
            highest = game.max_tile_exponent;
        }
    }
    return highest;
}

ExperimentMetrics ExperimentResult::metrics() const {
    ExperimentMetrics summary{};
    summary.game_count = games.size();
    if (games.empty()) {
        return summary;
    }

    std::vector<std::uint64_t> scores;
    scores.reserve(games.size());
    long double score_total = 0.0;
    long double max_tile_total = 0.0;
    long double move_total = 0.0;
    long double runtime_total = 0.0;
    std::uint64_t total_moves = 0;
    std::vector<std::uint64_t> move_counts;
    move_counts.reserve(games.size());
    summary.worst_score = std::numeric_limits<std::uint64_t>::max();

    for (const auto& game : games) {
        scores.push_back(game.score);
        move_counts.push_back(game.moves);
        score_total += static_cast<long double>(game.score);
        move_total += static_cast<long double>(game.moves);
        runtime_total += static_cast<long double>(game.runtime_seconds);
        total_moves += game.moves;
        summary.best_score = std::max(summary.best_score, game.score);
        summary.worst_score = std::min(summary.worst_score, game.score);

        const auto max_tile = game.max_tile_exponent == 0
            ? std::uint64_t{0}
            : std::uint64_t{1} << game.max_tile_exponent;
        max_tile_total += static_cast<long double>(max_tile);
        summary.highest_tile = std::max(summary.highest_tile, max_tile);
        ++summary.max_tile_distribution[game.max_tile_exponent];
        summary.achievement_rates.tile_1024 += game.max_tile_exponent >= 10 ? 1.0 : 0.0;
        summary.achievement_rates.tile_2048 += game.max_tile_exponent >= 11 ? 1.0 : 0.0;
        summary.achievement_rates.tile_4096 += game.max_tile_exponent >= 12 ? 1.0 : 0.0;
        summary.achievement_rates.tile_8192 += game.max_tile_exponent >= 13 ? 1.0 : 0.0;
        summary.achievement_rates.tile_16384 += game.max_tile_exponent >= 14 ? 1.0 : 0.0;
        summary.achievement_rates.tile_32768 += game.max_tile_exponent >= 15 ? 1.0 : 0.0;
        summary.achievement_rates.tile_65536 += game.max_tile_exponent >= 16 ? 1.0 : 0.0;
    }

    const auto count = static_cast<long double>(games.size());
    summary.mean_score = static_cast<double>(score_total / count);
    summary.mean_max_tile = static_cast<double>(max_tile_total / count);
    summary.mean_moves = static_cast<double>(move_total / count);
    summary.mean_runtime_seconds = static_cast<double>(runtime_total / count);
    summary.mean_milliseconds_per_move = total_moves == 0
        ? 0.0
        : static_cast<double>(runtime_total * 1'000.0L /
                              static_cast<long double>(total_moves));
    std::size_t mode_exponent = 0;
    for (std::size_t exponent = 1; exponent < summary.max_tile_distribution.size(); ++exponent) {
        // Prefer the higher tile when a small sample has tied modes.
        if (summary.max_tile_distribution[exponent] >= summary.max_tile_distribution[mode_exponent]) {
            mode_exponent = exponent;
        }
    }
    summary.mode_max_tile = mode_exponent == 0U
        ? std::uint64_t{0}
        : std::uint64_t{1} << mode_exponent;

    const auto rate_denominator = static_cast<double>(games.size());
    summary.achievement_rates.tile_1024 /= rate_denominator;
    summary.achievement_rates.tile_2048 /= rate_denominator;
    summary.achievement_rates.tile_4096 /= rate_denominator;
    summary.achievement_rates.tile_8192 /= rate_denominator;
    summary.achievement_rates.tile_16384 /= rate_denominator;
    summary.achievement_rates.tile_32768 /= rate_denominator;
    summary.achievement_rates.tile_65536 /= rate_denominator;

    std::sort(scores.begin(), scores.end());
    const auto middle = scores.size() / 2U;
    summary.median_score = scores.size() % 2U == 0U
        ? static_cast<double>((static_cast<long double>(scores[middle - 1U]) +
                               static_cast<long double>(scores[middle])) / 2.0L)
        : static_cast<double>(scores[middle]);

    std::sort(move_counts.begin(), move_counts.end());
    summary.median_moves = move_counts.size() % 2U == 0U
        ? static_cast<double>((static_cast<long double>(move_counts[middle - 1U]) +
                               static_cast<long double>(move_counts[middle])) / 2.0L)
        : static_cast<double>(move_counts[middle]);

    if (games.size() > 1U) {
        long double squared_difference_total = 0.0;
        for (const auto score : scores) {
            const auto difference = static_cast<long double>(score) - summary.mean_score;
            squared_difference_total += difference * difference;
        }
        summary.score_standard_deviation = static_cast<double>(std::sqrt(
            squared_difference_total / static_cast<long double>(games.size() - 1U)));
        const auto margin = 1.96 * summary.score_standard_deviation /
                            std::sqrt(static_cast<double>(games.size()));
        summary.score_confidence_95_low = summary.mean_score - margin;
        summary.score_confidence_95_high = summary.mean_score + margin;
    } else {
        summary.score_confidence_95_low = summary.mean_score;
        summary.score_confidence_95_high = summary.mean_score;
    }
    return summary;
}

ExperimentResult GameRunner::run(Agent& agent, RunConfig config) {
    if (config.game_count == 0) {
        throw std::invalid_argument("experiment must contain at least one game");
    }
    const auto additional_seeds = static_cast<std::uint64_t>(config.game_count - 1U);
    if (additional_seeds > std::numeric_limits<std::uint64_t>::max() - config.first_seed) {
        throw std::invalid_argument("experiment seed range exceeds uint64_t");
    }

    ExperimentResult experiment{
        std::string(agent.name()),
        config,
        {},
        0.0,
    };
    experiment.games.reserve(config.game_count);
    const auto experiment_start = std::chrono::steady_clock::now();

    for (std::size_t game_index = 0; game_index < config.game_count; ++game_index) {
        const auto game_start = std::chrono::steady_clock::now();
        const auto seed = config.first_seed + game_index;
        Game game(seed);
        agent.reset(decision_seed(seed));

        while (!game.game_over()) {
            const auto direction = agent.choose_move(game.board());
            if (!direction.has_value()) {
                throw std::logic_error("agent returned no move for a playable board");
            }
            const auto turn = game.apply_move(*direction);
            if (!turn.moved) {
                throw std::logic_error("agent selected an illegal move");
            }
        }

        const auto game_runtime = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - game_start).count();
        experiment.games.push_back(GameRecord{
            seed,
            game.score(),
            game.move_count(),
            max_exponent(game.board()),
            game_runtime,
        });
    }

    experiment.runtime_seconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - experiment_start).count();
    return experiment;
}

}  // namespace adversarial_2048
