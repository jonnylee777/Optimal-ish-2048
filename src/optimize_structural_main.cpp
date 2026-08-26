// Legacy CLI, superseded by run_experiment (src/run_experiment_main.cpp) —
// retained so old results in legacy/results/ remain reproducible.
#include "optimization/structural_optimizer.hpp"

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

[[nodiscard]] bool parse_probability(std::string_view text, double& value) {
    const std::string copy(text);
    char* end = nullptr;
    errno = 0;
    value = std::strtod(copy.c_str(), &end);
    return errno == 0 && end == copy.c_str() + copy.size() &&
           std::isfinite(value) && value >= 0.0 && value < 1.0;
}

void print_usage(std::string_view executable) {
    std::cerr
        << "Usage: " << executable << " [options]\n\n"
        << "Optimizes all four structural weights. By default, the optimized V1\n"
        << "baseline weights remain frozen for a controlled structural ablation.\n\n"
        << "Options:\n"
        << "  --optimize-baseline        Jointly optimize all four V1 baseline weights\n"
        << "  --candidates N             Candidate count (default: 24 frozen, 64 joint)\n"
        << "  --optimizer-seed N         Candidate-generation seed\n"
        << "  --depth N                  Player-layer search depth (default: 3)\n"
        << "  --first-training-seed N    First matched training seed (default: 1200)\n"
        << "  --probability-cutoff P     0 for exact search (default: 0)\n"
        << "  --help                     Show this message\n\n"
        << "Frozen stages: all candidates on 3 games, 8 survivors on 8 games,\n"
        << "then 3 finalists on 20 games. Joint stages use 16 and 6 survivors.\n"
        << "Results are written to results/.\n";
}

}  // namespace

int main(int argc, char* argv[]) {
    a2048::StructuralOptimizationConfig config;
    config.build_mode = ADVERSARIAL_2048_BUILD_TYPE;
    std::uint64_t candidate_count = config.candidate_count;
    std::uint64_t depth = config.search_depth;
    bool candidate_count_explicit = false;

    for (int index = 1; index < argc; ++index) {
        const std::string_view option(argv[index]);
        if (option == "--help") {
            print_usage(argv[0]);
            return 0;
        }
        if (option == "--optimize-baseline") {
            config.optimize_baseline_weights = true;
            continue;
        }
        if (index + 1 >= argc) {
            std::cerr << "Missing value for " << option << '\n';
            print_usage(argv[0]);
            return 2;
        }
        const std::string_view value(argv[++index]);
        bool valid = false;
        if (option == "--candidates") {
            valid = parse_uint64(value, candidate_count);
            candidate_count_explicit = valid;
        } else if (option == "--optimizer-seed") {
            valid = parse_uint64(value, config.optimizer_seed);
        } else if (option == "--depth") {
            valid = parse_uint64(value, depth) && depth > 0 &&
                    depth <= std::numeric_limits<std::uint32_t>::max();
        } else if (option == "--first-training-seed") {
            valid = parse_uint64(value, config.first_training_seed);
        } else if (option == "--probability-cutoff") {
            valid = parse_probability(value, config.minimum_path_probability);
        } else {
            std::cerr << "Unknown option: " << option << '\n';
            print_usage(argv[0]);
            return 2;
        }
        if (!valid) {
            std::cerr << "Invalid value for " << option << ": " << value << '\n';
            return 2;
        }
    }

    if (config.optimize_baseline_weights && !candidate_count_explicit) {
        candidate_count = 64;
    }
    const std::uint64_t minimum_candidates =
        config.optimize_baseline_weights ? 16U : 8U;
    if (candidate_count < minimum_candidates) {
        std::cerr << "Candidate count must be at least " << minimum_candidates
                  << (config.optimize_baseline_weights ? " in joint mode\n"
                                                       : " in frozen mode\n");
        return 2;
    }

    if (candidate_count > std::numeric_limits<std::size_t>::max()) {
        std::cerr << "Candidate count is out of range\n";
        return 2;
    }
    config.candidate_count = static_cast<std::size_t>(candidate_count);
    config.search_depth = static_cast<std::uint32_t>(depth);
    if (config.optimize_baseline_weights) {
        config.stages = {{3, 16}, {8, 6}, {20, 1}};
    }

    std::cout << "Structural weight optimization\n"
              << "mode="
              << (config.optimize_baseline_weights ? "joint 8-weight" : "structural-only")
              << '\n'
              << "depth=" << config.search_depth << '\n'
              << "candidates=" << config.candidate_count << '\n'
              << "training seeds begin at " << config.first_training_seed << '\n'
              << "probability cutoff=" << config.minimum_path_probability << '\n'
              << "initial baseline weights="
              << a2048::kDepth3OptimizedBaselineWeights.empty_cells << ','
              << a2048::kDepth3OptimizedBaselineWeights.monotonicity << ','
              << a2048::kDepth3OptimizedBaselineWeights.smoothness << ','
              << a2048::kDepth3OptimizedBaselineWeights.corner_preference << "\n\n";

    try {
        const auto result = a2048::optimize_structural_weights(
            config,
            [](const a2048::StructuralCandidateEvaluation& evaluation) {
                std::cout << "stage=" << evaluation.stage_index + 1
                          << " candidate=" << evaluation.candidate_id
                          << " games=" << evaluation.metrics.game_count
                          << " mean_score=" << std::fixed << std::setprecision(2)
                          << evaluation.metrics.mean_score
                          << " runtime=" << evaluation.runtime_seconds << "s\n"
                          << std::flush;
            });
        const auto files = a2048::write_structural_optimization_results(result);
        std::cout << std::setprecision(10)
                  << "best_candidate=" << result.best_candidate_id << '\n'
                  << "empty_cells=" << result.best_weights.baseline.empty_cells << '\n'
                  << "monotonicity=" << result.best_weights.baseline.monotonicity << '\n'
                  << "smoothness=" << result.best_weights.baseline.smoothness << '\n'
                  << "corner_preference="
                  << result.best_weights.baseline.corner_preference << '\n'
                  << "main_line=" << result.best_weights.main_line << '\n'
                  << "structural_stability="
                  << result.best_weights.structural_stability << '\n'
                  << "adverse_stuck=" << result.best_weights.adverse_stuck << '\n'
                  << "structural_displacement="
                  << result.best_weights.structural_displacement << '\n'
                  << "total_runtime_seconds=" << result.runtime_seconds << '\n'
                  << "CSV result: " << files.csv.string() << '\n'
                  << "JSON result: " << files.json.string() << '\n';
    } catch (const std::exception& error) {
        std::cerr << "Structural optimization failed: " << error.what() << '\n';
        return 1;
    }
}
