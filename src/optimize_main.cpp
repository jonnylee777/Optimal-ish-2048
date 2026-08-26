// Legacy CLI, superseded by run_experiment (src/run_experiment_main.cpp) —
// retained so old results in legacy/results/ remain reproducible.
#include "optimization/baseline_optimizer.hpp"

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
        << "Usage: " << executable << " [options]\n"
        << "\nOptions:\n"
        << "  --min-depth N              First depth to optimize (default: 3)\n"
        << "  --max-depth N              Last depth to optimize (default: 3)\n"
        << "  --candidates N             Candidate count, at least 6 (default: 16)\n"
        << "  --optimizer-seed N         Candidate-generation seed\n"
        << "  --probability-cutoff P     0 for exact search (default: 0)\n"
        << "  --help                     Show this message\n"
        << "\nLegacy usage remains supported:\n"
        << "  " << executable << " [candidate-count] [optimizer-seed]\n";
}

[[nodiscard]] bool starts_with_option(std::string_view value) {
    return value.starts_with("--");
}

}  // namespace

int main(int argc, char* argv[]) {
    a2048::BaselineOptimizationConfig base_config;
    base_config.build_mode = ADVERSARIAL_2048_BUILD_TYPE;
    std::uint64_t minimum_depth = base_config.search_depth;
    std::uint64_t maximum_depth = base_config.search_depth;
    std::uint64_t candidate_count = base_config.candidate_count;

    if (argc >= 2 && !starts_with_option(argv[1])) {
        if (argc > 3 || !parse_uint64(argv[1], candidate_count) || candidate_count < 6 ||
            (argc == 3 && !parse_uint64(argv[2], base_config.optimizer_seed))) {
            print_usage(argv[0]);
            return 2;
        }
    } else {
        for (int index = 1; index < argc; ++index) {
            const std::string_view option(argv[index]);
            if (option == "--help") {
                print_usage(argv[0]);
                return 0;
            }
            if (index + 1 >= argc) {
                std::cerr << "Missing value for " << option << '\n';
                print_usage(argv[0]);
                return 2;
            }
            const std::string_view value(argv[++index]);
            bool valid = false;
            if (option == "--min-depth") {
                valid = parse_uint64(value, minimum_depth) && minimum_depth > 0;
            } else if (option == "--max-depth") {
                valid = parse_uint64(value, maximum_depth) && maximum_depth > 0;
            } else if (option == "--candidates") {
                valid = parse_uint64(value, candidate_count) && candidate_count >= 6;
            } else if (option == "--optimizer-seed") {
                valid = parse_uint64(value, base_config.optimizer_seed);
            } else if (option == "--probability-cutoff") {
                valid = parse_probability(value, base_config.minimum_path_probability);
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
    }

    if (minimum_depth > maximum_depth ||
        maximum_depth > std::numeric_limits<std::uint32_t>::max() ||
        candidate_count > std::numeric_limits<std::size_t>::max()) {
        std::cerr << "Depth range or candidate count is out of range\n";
        return 2;
    }
    base_config.candidate_count = static_cast<std::size_t>(candidate_count);
    base_config.stages[0].survivor_count = 6;

    std::cout << "Optimizing depths " << minimum_depth << '-' << maximum_depth
              << " with " << base_config.candidate_count << " candidates and "
              << (base_config.minimum_path_probability == 0.0 ? "exact" : "approximate")
              << " Expectimax\n"
              << "Results are written to results/ after each depth completes.\n";

    try {
        for (auto depth = minimum_depth; depth <= maximum_depth; ++depth) {
            auto config = base_config;
            config.search_depth = static_cast<std::uint32_t>(depth);
            std::cout << "\nStarting depth " << depth << "\n" << std::flush;
            const auto result = a2048::optimize_baseline_weights(
                config,
                [depth](const a2048::CandidateEvaluation& evaluation) {
                    std::cout << "depth=" << depth
                              << " stage=" << evaluation.stage_index + 1
                              << " candidate=" << evaluation.candidate_id
                              << " games=" << evaluation.metrics.game_count
                              << " mean_score=" << std::fixed << std::setprecision(2)
                              << evaluation.metrics.mean_score
                              << " runtime=" << evaluation.runtime_seconds << "s\n"
                              << std::flush;
                });
            const auto files = a2048::write_optimization_results(result);
            std::cout << std::setprecision(10)
                      << "Completed depth " << depth << '\n'
                      << "best_candidate=" << result.best_candidate_id << '\n'
                      << "empty_cells=" << result.best_weights.empty_cells << '\n'
                      << "monotonicity=" << result.best_weights.monotonicity << '\n'
                      << "smoothness=" << result.best_weights.smoothness << '\n'
                      << "corner_preference=" << result.best_weights.corner_preference << '\n'
                      << "total_runtime_seconds=" << result.runtime_seconds << '\n'
                      << "CSV result: " << files.csv.string() << '\n'
                      << "JSON result: " << files.json.string() << '\n';
        }
    } catch (const std::exception& error) {
        std::cerr << "Optimization failed: " << error.what() << '\n';
        return 1;
    }
}
