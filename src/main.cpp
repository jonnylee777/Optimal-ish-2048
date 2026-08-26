// Legacy CLI, superseded by run_experiment (src/run_experiment_main.cpp) —
// retained so old results in legacy/results/ remain reproducible.
#include "agents/greedy_agent.hpp"
#include "agents/random_agent.hpp"
#include "agents/search_agent.hpp"
#include "evaluation/baseline_heuristic.hpp"
#include "evaluation/structural_heuristic.hpp"
#include "experiments/game_runner.hpp"
#include "experiments/result_writer.hpp"
#include "experiments/seed_sets.hpp"

#include <charconv>
#include <cerrno>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <system_error>

namespace a2048 = adversarial_2048;

#ifndef ADVERSARIAL_2048_BUILD_TYPE
#define ADVERSARIAL_2048_BUILD_TYPE "unknown"
#endif

namespace {

[[nodiscard]] bool parse_uint64(std::string_view text, std::uint64_t& value) {
    const auto result = std::from_chars(text.data(), text.data() + text.size(), value);
    return result.ec == std::errc{} && result.ptr == text.data() + text.size();
}

[[nodiscard]] bool parse_nonnegative_double(std::string_view text, double& value) {
    const std::string copy(text);
    char* end = nullptr;
    errno = 0;
    value = std::strtod(copy.c_str(), &end);
    return errno == 0 && end == copy.c_str() + copy.size() &&
           std::isfinite(value) && value >= 0.0;
}

void print_usage(std::string_view executable) {
    std::cerr << "Usage: " << executable
              << " <random|greedy|baseline|baseline-optimized"
                 "|structural-mainline|structural-movement|structural-full>"
                 " [games] [first-seed] [depth] [probability-cutoff]\n"
              << "   or: " << executable
              << " <v1|v1.1|v2|v2.1> [games] [first-seed]"
                 " [depth=8] [probability-cutoff] [time-limit-ms]\n"
              << "       depth 8 uses adaptive 4/6/8; other depths are fixed\n"
              << "   or: " << executable
              << " baseline-custom <games> <first-seed> <depth>"
                 " <empty> <monotonicity> <smoothness> <corner>"
                 " [probability-cutoff]\n"
              << "   or: " << executable
              << " structural-custom <games> <first-seed> <depth>"
                 " <empty> <monotonicity> <smoothness> <corner>"
                 " <main-line> <stability> <adverse-stuck> <displacement>"
                 " [probability-cutoff]\n";
}

}  // namespace

int main(int argc, char* argv[]) {
    if (argc < 2) {
        print_usage(argv[0]);
        return 2;
    }

    const std::string_view agent_name(argv[1]);
    const bool custom_baseline = agent_name == "baseline-custom";
    const bool custom_structural = agent_name == "structural-custom";
    const bool adaptive_v1 = agent_name == "v1";
    const bool adaptive_v1_1 = agent_name == "v1.1";
    const bool adaptive_v2 = agent_name == "v2";
    const bool adaptive_v2_1 = agent_name == "v2.1";
    const bool versioned_policy =
        adaptive_v1 || adaptive_v1_1 || adaptive_v2 || adaptive_v2_1;
    const bool optimized_baseline =
        agent_name == "baseline-optimized" || adaptive_v1_1;
    const bool named_baseline =
        agent_name == "baseline" || agent_name == "baseline-optimized" ||
        adaptive_v1 || adaptive_v1_1;
    const bool structural_mainline = agent_name == "structural-mainline";
    const bool structural_movement = agent_name == "structural-movement";
    const bool structural_full = agent_name == "structural-full";
    const bool structural_policy =
        structural_mainline || structural_movement || structural_full || custom_structural ||
        adaptive_v2 || adaptive_v2_1;
    const bool search_policy = named_baseline || custom_baseline || structural_policy;
    if ((!custom_baseline && !custom_structural && !versioned_policy && argc > 6) ||
        (versioned_policy && argc > 7) ||
        (custom_baseline && argc != 9 && argc != 10) ||
        (custom_structural && argc != 13 && argc != 14)) {
        print_usage(argv[0]);
        return 2;
    }
    std::uint64_t game_count = 100;
    std::uint64_t first_seed = 1'000;
    std::uint64_t depth_value = versioned_policy ? 8U : 2U;
    if ((argc >= 3 && !parse_uint64(argv[2], game_count)) || game_count == 0 ||
        game_count > std::numeric_limits<std::size_t>::max() ||
        (argc >= 4 && !parse_uint64(argv[3], first_seed)) ||
        (argc >= 5 && !parse_uint64(argv[4], depth_value)) || depth_value == 0 ||
        depth_value > std::numeric_limits<std::uint32_t>::max() ||
        (argc >= 5 && !search_policy)) {
        print_usage(argv[0]);
        return 2;
    }
    const bool adaptive_policy = versioned_policy && depth_value == 8U;

    a2048::BaselineWeights baseline_weights;
    if (optimized_baseline) {
        baseline_weights = a2048::kDepth3OptimizedBaselineWeights;
    }
    if (custom_baseline &&
        (!parse_nonnegative_double(argv[5], baseline_weights.empty_cells) ||
         !parse_nonnegative_double(argv[6], baseline_weights.monotonicity) ||
         !parse_nonnegative_double(argv[7], baseline_weights.smoothness) ||
         !parse_nonnegative_double(argv[8], baseline_weights.corner_preference))) {
        print_usage(argv[0]);
        return 2;
    }
    auto structural_weights = structural_mainline
        ? a2048::main_line_ablation_weights()
        : (structural_movement
               ? a2048::movement_ablation_weights()
               : a2048::StructuralWeights{});
    if (adaptive_v2_1) {
        structural_weights = a2048::kDepth4OptimizedStructuralWeights;
    }
    if (custom_structural &&
        (!parse_nonnegative_double(argv[5], structural_weights.baseline.empty_cells) ||
         !parse_nonnegative_double(argv[6], structural_weights.baseline.monotonicity) ||
         !parse_nonnegative_double(argv[7], structural_weights.baseline.smoothness) ||
         !parse_nonnegative_double(
             argv[8], structural_weights.baseline.corner_preference) ||
         !parse_nonnegative_double(argv[9], structural_weights.main_line) ||
         !parse_nonnegative_double(
             argv[10], structural_weights.structural_stability) ||
         !parse_nonnegative_double(argv[11], structural_weights.adverse_stuck) ||
         !parse_nonnegative_double(
             argv[12], structural_weights.structural_displacement))) {
        print_usage(argv[0]);
        return 2;
    }
    double minimum_path_probability = 0.0;
    const auto cutoff_argument = custom_structural ? 13 : (custom_baseline ? 9 : 5);
    if (argc > cutoff_argument &&
        (!parse_nonnegative_double(argv[cutoff_argument], minimum_path_probability) ||
         minimum_path_probability >= 1.0)) {
        print_usage(argv[0]);
        return 2;
    }
    double time_limit_seconds = adaptive_policy ? 1.0 : 0.0;
    if (versioned_policy && argc > 6) {
        double time_limit_milliseconds = 0.0;
        if (!parse_nonnegative_double(argv[6], time_limit_milliseconds)) {
            print_usage(argv[0]);
            return 2;
        }
        time_limit_seconds = time_limit_milliseconds / 1'000.0;
    }
    a2048::BaselineHeuristic baseline_evaluator(baseline_weights);
    a2048::StructuralHeuristic structural_evaluator(structural_weights);
    std::unique_ptr<a2048::Agent> agent;
    a2048::SearchAgent* search_agent = nullptr;
    if (agent_name == "random") {
        agent = std::make_unique<a2048::RandomAgent>();
    } else if (agent_name == "greedy") {
        agent = std::make_unique<a2048::GreedyAgent>();
    } else if (search_policy) {
        const a2048::Evaluator& evaluator = structural_policy
            ? static_cast<const a2048::Evaluator&>(structural_evaluator)
            : static_cast<const a2048::Evaluator&>(baseline_evaluator);
        auto baseline_agent = adaptive_policy
            ? std::make_unique<a2048::SearchAgent>(
                  evaluator,
                  a2048::AdaptiveDepthSchedule{4, 6, 8},
                  std::string(agent_name),
                  a2048::ExpectimaxOptions{
                      true,
                      1U << 16U,
                      minimum_path_probability,
                      false,
                      time_limit_seconds})
            : std::make_unique<a2048::SearchAgent>(
                  evaluator,
                  static_cast<std::uint32_t>(depth_value),
                  std::string(agent_name),
                  a2048::ExpectimaxOptions{
                      true,
                      1U << 16U,
                      minimum_path_probability,
                      false,
                      time_limit_seconds});
        search_agent = baseline_agent.get();
        agent = std::move(baseline_agent);
    } else {
        print_usage(argv[0]);
        return 2;
    }

    try {
        const auto result = a2048::GameRunner::run(
            *agent,
            {static_cast<std::size_t>(game_count), first_seed});
        const auto metrics = result.metrics();
        const auto last_seed = result.games.back().seed;

        std::string evaluator_type{"none"};
        std::string feature_configuration{"none"};
        std::string optimization_configuration{"none"};
        if (structural_policy) {
            evaluator_type = adaptive_v2_1
                ? "structural-heuristic-optimized-v2.1"
                : (custom_structural ? "structural-heuristic-custom"
                                     : "structural-heuristic-initial-v2");
            if (adaptive_v2_1) {
                feature_configuration =
                    "v2.1:jointly-optimized-four-part-baseline+all-structural-features";
                optimization_configuration =
                    "exact depth-4 joint optimization; candidate 6; training seeds 1200-1219";
            } else if (adaptive_v2) {
                feature_configuration =
                    "v2:optimized-four-part-baseline+initial-structural-features";
                optimization_configuration =
                    "initial experimental structural weights; baseline terms use depth-3 optimized weights";
            } else if (custom_structural) {
                feature_configuration = "structural-v1:custom-eight-weight-vector";
                optimization_configuration =
                    "custom evaluator parameters supplied on command line";
            } else if (structural_mainline) {
                feature_configuration =
                    "structural-v1:baseline-optimized+main-line";
                optimization_configuration =
                    "initial experimental structural weights; baseline terms use depth-3 optimized weights";
            } else if (structural_movement) {
                feature_configuration =
                    "structural-v1:baseline-optimized+main-line+displacement";
                optimization_configuration =
                    "initial experimental structural weights; baseline terms use depth-3 optimized weights";
            } else {
                feature_configuration =
                    "structural-v1:baseline-optimized+main-line+displacement+stability+adverse-stuck";
                optimization_configuration =
                    "initial experimental structural weights; baseline terms use depth-3 optimized weights";
            }
        } else if (named_baseline || custom_baseline) {
            evaluator_type = optimized_baseline
                ? "baseline-heuristic-optimized-v1.1"
                : "baseline-heuristic-v1";
            feature_configuration = adaptive_v1_1
                ? "v1.1:optimized-empty-cells,monotonicity,smoothness,corner-preference"
                : "v1:empty-cells,monotonicity,smoothness,corner-preference";
            optimization_configuration = optimized_baseline
                ? "depth-3 deterministic multistage random search; candidate 9"
                : "none";
        }

        a2048::ResultMetadata metadata{
            evaluator_type,
            {},
            feature_configuration,
            search_agent == nullptr ? 0U : search_agent->depth(),
            search_agent == nullptr
                ? "not applicable"
                : (adaptive_policy
                       ? "adaptive player-layer depth: 10-16 empty=4, 6-9 empty=6, 0-5 empty=8"
                       : "depth counts player layers; depth 1 = player move, spawn, evaluator"),
            std::string(a2048::seed_sets::classify(first_seed, last_seed)),
            ADVERSARIAL_2048_BUILD_TYPE,
            "unavailable",
            optimization_configuration,
            std::nullopt,
            minimum_path_probability,
            std::nullopt,
            time_limit_seconds,
            std::nullopt,
        };
        if (search_agent != nullptr && search_agent->uses_adaptive_depth()) {
            metadata.adaptive_depth_usage = search_agent->adaptive_depth_usage();
            metadata.completed_depth_usage = search_agent->completed_depth_usage();
        }
        if (structural_policy) {
            metadata.evaluator_parameters = {
                {"empty_cells", structural_weights.baseline.empty_cells},
                {"monotonicity", structural_weights.baseline.monotonicity},
                {"smoothness", structural_weights.baseline.smoothness},
                {"corner_preference", structural_weights.baseline.corner_preference},
                {"main_line", structural_weights.main_line},
                {"structural_stability", structural_weights.structural_stability},
                {"adverse_stuck", structural_weights.adverse_stuck},
                {"structural_displacement", structural_weights.structural_displacement},
            };
        } else if (named_baseline || custom_baseline) {
            const auto& weights = baseline_evaluator.weights();
            metadata.evaluator_parameters = {
                {"empty_cells", weights.empty_cells},
                {"monotonicity", weights.monotonicity},
                {"smoothness", weights.smoothness},
                {"corner_preference", weights.corner_preference},
            };
        }
        if (search_agent != nullptr) {
            metadata.search_statistics = search_agent->cumulative_statistics();
        }
        const auto files = a2048::write_experiment_results(result, metadata);

        std::cout << std::fixed << std::setprecision(2)
                  << "agent:          " << result.agent_name << '\n'
                  << "games:          " << result.games.size() << '\n'
                  << "seed range:     " << result.config.first_seed << '-'
                  << last_seed << " (" << metadata.seed_partition << ")\n"
                  << "mean score:     " << metrics.mean_score << '\n'
                  << "median score:   " << metrics.median_score << '\n'
                  << "score stddev:   " << metrics.score_standard_deviation << '\n'
                  << "95% CI:         [" << metrics.score_confidence_95_low << ", "
                  << metrics.score_confidence_95_high << "]\n"
                  << "best score:     " << metrics.best_score << '\n'
                  << "mean max tile:  " << metrics.mean_max_tile << '\n'
                  << "highest tile:   " << metrics.highest_tile << '\n'
                  << "1024/2048:      " << metrics.achievement_rates.tile_1024 * 100.0
                  << "% / " << metrics.achievement_rates.tile_2048 * 100.0 << "%\n"
                  << "4096/8192:      " << metrics.achievement_rates.tile_4096 * 100.0
                  << "% / " << metrics.achievement_rates.tile_8192 * 100.0 << "%\n"
                  << "mean moves:     " << metrics.mean_moves << '\n'
                  << "runtime (sec):  " << std::setprecision(6)
                  << result.runtime_seconds << '\n';

        if (search_agent != nullptr) {
            const auto& statistics = search_agent->cumulative_statistics();
            std::uint64_t moves_searched = 0;
            for (const auto& game : result.games) {
                moves_searched += game.moves;
            }
            const auto milliseconds_per_move = moves_searched == 0
                ? 0.0
                : statistics.elapsed_seconds * 1'000.0 /
                    static_cast<double>(moves_searched);
            std::cout << std::setprecision(2)
                      << "search depth:   " << search_agent->depth() << '\n'
                      << "depth mode:     "
                      << (search_agent->uses_adaptive_depth() ? "adaptive" : "fixed")
                      << '\n'
                      << "prob. cutoff:   " << std::setprecision(6)
                      << search_agent->minimum_path_probability() << '\n'
                      << "time/move cap:  "
                      << search_agent->time_limit_seconds() * 1'000.0 << " ms\n"
                      << "search nodes:   " << statistics.total_nodes() << '\n'
                      << "player nodes:   " << statistics.player_nodes << '\n'
                      << "chance nodes:   " << statistics.chance_nodes << '\n'
                      << "leaf evals:     " << statistics.leaf_evaluations << '\n'
                      << "nodes/sec:      " << statistics.nodes_per_second() << '\n'
                      << "ms/move:        " << std::setprecision(4)
                      << milliseconds_per_move << '\n'
                      << std::setprecision(2)
                      << "cache hit rate: " << statistics.cache_hit_rate() * 100.0 << "%\n";
            if (search_agent->uses_adaptive_depth()) {
                const auto& usage = search_agent->adaptive_depth_usage();
                std::cout << "depth-4 moves:  " << usage[0] << '\n'
                          << "depth-6 moves:  " << usage[1] << '\n'
                          << "depth-8 moves:  " << usage[2] << '\n';
                const auto& completed = search_agent->completed_depth_usage();
                std::cout << "completed depth:";
                for (std::size_t depth = 0; depth < completed.size(); ++depth) {
                    if (completed[depth] != 0) {
                        std::cout << ' ' << depth << '=' << completed[depth];
                    }
                }
                std::cout << '\n';
            }
        }
        std::cout << "CSV result:     " << files.csv.string() << '\n'
                  << "JSON result:    " << files.json.string() << '\n';
    } catch (const std::exception& error) {
        std::cerr << "Experiment failed: " << error.what() << '\n';
        return 1;
    }
}
