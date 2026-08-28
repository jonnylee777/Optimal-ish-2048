// Standardized experiment runner. Flag-based configuration keeps heuristic
// choice, weight overrides, and search configuration independent instead of
// coupling them into one opaque policy name — see docs/phase1-heuristics.md.
#include "agents/search_agent.hpp"
#include "evaluation/baseline_heuristic.hpp"
#include "evaluation/h0_heuristic.hpp"
#include "evaluation/h2_heuristic.hpp"
#include "evaluation/h3_heuristic.hpp"
#include "evaluation/h4_heuristic.hpp"
#include "evaluation/h5_heuristic.hpp"
#include "evaluation/n1_evaluator.hpp"
#include "learning/ntuple_network.hpp"
#include "experiments/game_runner.hpp"
#include "experiments/result_writer.hpp"
#include "experiments/run_experiment_config.hpp"

#include <array>
#include <cstdint>
#include <cstdio>
#include <exception>
#include <iostream>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#ifndef ADVERSARIAL_2048_BUILD_TYPE
#define ADVERSARIAL_2048_BUILD_TYPE "unknown"
#endif

namespace a2048 = adversarial_2048;

namespace {

void print_usage(const char* program) {
    std::cerr
        << "Usage: " << program << " \\\n"
        << "    --heuristic H0|H1|H2|H3|H4|H5|N1 [--weight name=value ...] \\\n"
        << "    [--weights PATH]      # required for N1 (learned evaluator)\n"
        << "    --search fixed|timed \\\n"
        << "    [--depth N] [--time-limit-ms N] [--adaptive-schedule H,M,L] \\\n"
        << "    [--transposition-table on|off] [--tt-capacity N] \\\n"
        << "    [--probability-cutoff X] [--symmetry on|off] \\\n"
        << "    --seeds quick|standard|final|FIRST-LAST [--games N] \\\n"
        << "    [--output-dir DIR] [--quiet]\n";
}

[[nodiscard]] a2048::BaselineWeights resolve_h1_weights(
    const std::vector<a2048::NamedParameter>& overrides) {
    a2048::BaselineWeights weights{};
    for (const auto& parameter : overrides) {
        if (parameter.name == "empty_cells") {
            weights.empty_cells = parameter.value;
        } else if (parameter.name == "monotonicity") {
            weights.monotonicity = parameter.value;
        } else if (parameter.name == "smoothness") {
            weights.smoothness = parameter.value;
        } else if (parameter.name == "corner_preference") {
            weights.corner_preference = parameter.value;
        } else {
            throw std::invalid_argument("H1 has no weight named '" + parameter.name + "'");
        }
    }
    return weights;
}

[[nodiscard]] a2048::H2Weights resolve_h2_weights(
    const std::vector<a2048::NamedParameter>& overrides) {
    a2048::H2Weights weights{};
    for (const auto& parameter : overrides) {
        if (parameter.name == "empty_cells") {
            weights.baseline.empty_cells = parameter.value;
        } else if (parameter.name == "monotonicity") {
            weights.baseline.monotonicity = parameter.value;
        } else if (parameter.name == "smoothness") {
            weights.baseline.smoothness = parameter.value;
        } else if (parameter.name == "corner_preference") {
            weights.baseline.corner_preference = parameter.value;
        } else if (parameter.name == "corner_chain") {
            weights.corner_chain = parameter.value;
        } else {
            throw std::invalid_argument("H2 has no weight named '" + parameter.name + "'");
        }
    }
    return weights;
}

[[nodiscard]] a2048::H4Weights resolve_h4_weights(
    const std::vector<a2048::NamedParameter>& overrides) {
    a2048::H4Weights weights{};
    for (const auto& parameter : overrides) {
        if (parameter.name == "empty") {
            weights.empty = parameter.value;
        } else if (parameter.name == "merges") {
            weights.merges = parameter.value;
        } else if (parameter.name == "monotonicity") {
            weights.monotonicity = parameter.value;
        } else if (parameter.name == "sum") {
            weights.sum = parameter.value;
        } else {
            throw std::invalid_argument("H4 has no weight named '" + parameter.name + "'");
        }
    }
    return weights;
}

[[nodiscard]] a2048::H3Weights resolve_h3_weights(
    const std::vector<a2048::NamedParameter>& overrides) {
    a2048::H3Weights weights{};
    for (const auto& parameter : overrides) {
        if (parameter.name == "empty_cells") {
            weights.h2.baseline.empty_cells = parameter.value;
        } else if (parameter.name == "monotonicity") {
            weights.h2.baseline.monotonicity = parameter.value;
        } else if (parameter.name == "smoothness") {
            weights.h2.baseline.smoothness = parameter.value;
        } else if (parameter.name == "corner_preference") {
            weights.h2.baseline.corner_preference = parameter.value;
        } else if (parameter.name == "corner_chain") {
            weights.h2.corner_chain = parameter.value;
        } else if (parameter.name == "main_line") {
            weights.main_line = parameter.value;
        } else if (parameter.name == "structural_stability") {
            weights.structural_stability = parameter.value;
        } else if (parameter.name == "adverse_stuck") {
            weights.adverse_stuck = parameter.value;
        } else if (parameter.name == "structural_displacement") {
            weights.structural_displacement = parameter.value;
        } else {
            throw std::invalid_argument("H3 has no weight named '" + parameter.name + "'");
        }
    }
    return weights;
}

[[nodiscard]] a2048::H0Weights resolve_h0_weights(
    const std::vector<a2048::NamedParameter>& overrides) {
    a2048::H0Weights weights{};
    for (const auto& parameter : overrides) {
        if (parameter.name == "empty_cells") {
            weights.empty_cells = parameter.value;
        } else if (parameter.name == "edge_max_bonus") {
            weights.edge_max_bonus = parameter.value;
        } else {
            throw std::invalid_argument("H0 has no weight named '" + parameter.name + "'");
        }
    }
    return weights;
}

[[nodiscard]] std::string detect_git_commit() {
    FILE* pipe = popen("git rev-parse --short HEAD 2>/dev/null", "r");
    if (pipe == nullptr) {
        return "unavailable";
    }
    std::string result;
    std::array<char, 64> buffer{};
    while (std::fgets(buffer.data(), static_cast<int>(buffer.size()), pipe) != nullptr) {
        result += buffer.data();
    }
    const auto status = pclose(pipe);
    while (!result.empty() && (result.back() == '\n' || result.back() == '\r')) {
        result.pop_back();
    }
    return (status == 0 && !result.empty()) ? result : "unavailable";
}

int run(
    const a2048::RunExperimentConfig& config,
    const a2048::Evaluator& evaluator,
    const std::string& evaluator_type,
    const std::string& feature_configuration,
    const std::vector<a2048::NamedParameter>& evaluator_parameters) {
    if (config.symmetry && !evaluator.is_rotation_invariant()) {
        throw std::invalid_argument(
            evaluator_type + " is not rotation-invariant; --symmetry on is unsafe for it "
            "(see Evaluator::is_rotation_invariant in docs/phase1-heuristics.md)");
    }

    const a2048::ExpectimaxOptions options{
        .use_transposition_table = config.transposition_table,
        .transposition_table_capacity = config.transposition_table_capacity,
        .minimum_path_probability = config.probability_cutoff,
        .use_symmetry_reduction = config.symmetry,
        .time_limit_seconds = config.time_limit_ms / 1000.0,
    };

    const auto search_name = evaluator_type + (config.search == a2048::SearchMode::fixed
        ? "-fixed" : "-timed");
    std::unique_ptr<a2048::SearchAgent> agent =
        config.use_adaptive_schedule
        ? std::make_unique<a2048::SearchAgent>(
              evaluator,
              a2048::AdaptiveDepthSchedule{
                  config.adaptive_depths.high_empty_depth,
                  config.adaptive_depths.medium_empty_depth,
                  config.adaptive_depths.low_empty_depth,
              },
              search_name, options)
        : std::make_unique<a2048::SearchAgent>(evaluator, config.fixed_depth, search_name, options);

    if (!config.quiet) {
        std::cout << "running " << config.run_config.game_count << " games ("
                  << config.seed_set_label << ", seeds " << config.run_config.first_seed
                  << '-' << (config.run_config.first_seed + config.run_config.game_count - 1)
                  << ")...\n";
    }

    const auto result = a2048::GameRunner::run(*agent, config.run_config);

    std::optional<double> deadline_hit_rate;
    if (!config.use_adaptive_schedule) {
        const auto ceiling = config.fixed_depth;
        const auto& usage = agent->completed_depth_usage();
        std::uint64_t total = 0;
        for (const auto count : usage) {
            total += count;
        }
        if (total > 0 && ceiling < usage.size()) {
            deadline_hit_rate = 1.0 - static_cast<double>(usage[ceiling]) / static_cast<double>(total);
        }
    }

    const a2048::ResultMetadata metadata{
        .evaluator_type = evaluator_type,
        .evaluator_parameters = evaluator_parameters,
        .feature_configuration = feature_configuration,
        .search_depth = agent->depth(),
        .depth_definition = config.use_adaptive_schedule
            ? "adaptive player-layer depth: 10-16 empty=" +
                  std::to_string(config.adaptive_depths.high_empty_depth) + ", 6-9 empty=" +
                  std::to_string(config.adaptive_depths.medium_empty_depth) + ", 0-5 empty=" +
                  std::to_string(config.adaptive_depths.low_empty_depth)
            : "depth counts player layers; depth 1 = player move, spawn, evaluator",
        .seed_partition = config.seed_set_label,
        .build_mode = ADVERSARIAL_2048_BUILD_TYPE,
        .git_commit = detect_git_commit(),
        .optimization_configuration = "none",
        .search_statistics = agent->cumulative_statistics(),
        .minimum_path_probability = config.probability_cutoff,
        .adaptive_depth_usage = config.use_adaptive_schedule
            ? std::optional(agent->adaptive_depth_usage()) : std::nullopt,
        .time_limit_seconds = config.time_limit_ms / 1000.0,
        .completed_depth_usage = agent->completed_depth_usage(),
        .deadline_hit_rate = deadline_hit_rate,
    };

    const auto files = a2048::write_experiment_results(result, metadata, config.output_dir);
    if (!config.quiet) {
        const auto metrics = result.metrics();
        std::cout << "games: " << metrics.game_count
                  << "  mean score: " << metrics.mean_score
                  << "  median score: " << metrics.median_score
                  << "  min/max score: " << metrics.worst_score << '/' << metrics.best_score
                  << "  highest tile: " << metrics.highest_tile
                  << "  mean ms/move: " << metrics.mean_milliseconds_per_move << '\n'
                  << "wrote " << files.json.string() << " and " << files.csv.string() << '\n';
    }
    return 0;
}

}  // namespace

int main(int argc, char* argv[]) {
    const std::vector<std::string> args(argv + 1, argv + argc);
    a2048::RunExperimentConfig config;
    try {
        config = a2048::parse_run_experiment_args(args);
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << '\n';
        print_usage(argv[0]);
        return 2;
    }

    try {
        if (config.heuristic == a2048::HeuristicChoice::h0) {
            const a2048::H0Heuristic evaluator(resolve_h0_weights(config.weight_overrides));
            const auto& weights = evaluator.weights();
            return run(config, evaluator, "H0", "H0:empty_cells+edge_max_bonus",
                {{"empty_cells", weights.empty_cells}, {"edge_max_bonus", weights.edge_max_bonus}});
        }
        if (config.heuristic == a2048::HeuristicChoice::h2) {
            const a2048::H2Heuristic evaluator(resolve_h2_weights(config.weight_overrides));
            const auto& weights = evaluator.weights();
            return run(config, evaluator, "H2",
                "H2:empty_cells+monotonicity+smoothness+corner_preference+corner_chain",
                {{"empty_cells", weights.baseline.empty_cells},
                 {"monotonicity", weights.baseline.monotonicity},
                 {"smoothness", weights.baseline.smoothness},
                 {"corner_preference", weights.baseline.corner_preference},
                 {"corner_chain", weights.corner_chain}});
        }
        if (config.heuristic == a2048::HeuristicChoice::n1) {
            if (!config.weight_overrides.empty()) {
                throw std::invalid_argument(
                    "N1's weights are learned, not set on the command line; use --weights");
            }
            // The network is large (~128 MB) and is held by reference all the
            // way down into the search, so it must outlive the run() call.
            // Built from the file's own tuple definitions, so any trained
            // network shape plays without this binary needing to know which.
            auto network = a2048::learning::NTupleNetwork::load_from(config.weights_path);
            const a2048::N1Evaluator evaluator(network);
            // 33.7M weights cannot go in evaluator_parameters (name -> double),
            // so provenance is the file path plus a content fingerprint.
            return run(config, evaluator, "N1",
                "N1:ntuple-td-learned:" + config.weights_path.string() + ":" +
                    network.fingerprint(),
                {});
        }
        if (config.heuristic == a2048::HeuristicChoice::h5) {
            if (!config.weight_overrides.empty()) {
                throw std::invalid_argument(
                    "H5 has no configurable weights (it is a direct port of the "
                    "reference project's own fixed, already-tuned constants)");
            }
            const a2048::H5Heuristic evaluator;
            return run(config, evaluator, "H5",
                "H5:formation-aware-port:saturating-tile-weights+dpdf-vs-t-per-axis-max",
                {});
        }
        if (config.heuristic == a2048::HeuristicChoice::h4) {
            const a2048::H4Heuristic evaluator(resolve_h4_weights(config.weight_overrides));
            const auto& weights = evaluator.weights();
            return run(config, evaluator, "H4",
                "H4:nneonneo-reference:empty+merges+monotonicity+sum",
                {{"empty", weights.empty},
                 {"merges", weights.merges},
                 {"monotonicity", weights.monotonicity},
                 {"sum", weights.sum}});
        }
        if (config.heuristic == a2048::HeuristicChoice::h3) {
            const a2048::H3Heuristic evaluator(resolve_h3_weights(config.weight_overrides));
            const auto& weights = evaluator.weights();
            return run(config, evaluator, "H3",
                "H3:empty_cells+monotonicity+smoothness+corner_preference+corner_chain"
                "+main_line+structural_stability+adverse_stuck+structural_displacement",
                {{"empty_cells", weights.h2.baseline.empty_cells},
                 {"monotonicity", weights.h2.baseline.monotonicity},
                 {"smoothness", weights.h2.baseline.smoothness},
                 {"corner_preference", weights.h2.baseline.corner_preference},
                 {"corner_chain", weights.h2.corner_chain},
                 {"main_line", weights.main_line},
                 {"structural_stability", weights.structural_stability},
                 {"adverse_stuck", weights.adverse_stuck},
                 {"structural_displacement", weights.structural_displacement}});
        }
        const a2048::BaselineHeuristic evaluator(resolve_h1_weights(config.weight_overrides));
        const auto& weights = evaluator.weights();
        return run(config, evaluator, "H1",
            "H1:empty_cells+monotonicity+smoothness+corner_preference",
            {{"empty_cells", weights.empty_cells},
             {"monotonicity", weights.monotonicity},
             {"smoothness", weights.smoothness},
             {"corner_preference", weights.corner_preference}});
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << '\n';
        return 1;
    }
}
