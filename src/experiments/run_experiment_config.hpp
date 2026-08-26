#pragma once

#include "experiments/game_runner.hpp"
#include "experiments/result_writer.hpp"

#include <cstdint>
#include <filesystem>
#include <optional>
#include <string>
#include <vector>

namespace adversarial_2048 {

// Flag-based configuration for the standardized `run_experiment` CLI.
// See docs/experiment-taxonomy.md. Deliberately keeps heuristic choice,
// weight overrides, and search configuration as independent fields so no
// single flag can silently couple them together.
enum class HeuristicChoice { h0, h1, h2, h3, h4, h5, n1 };
enum class SearchMode { fixed, timed };

struct AdaptiveDepths {
    std::uint32_t high_empty_depth{4};
    std::uint32_t medium_empty_depth{6};
    std::uint32_t low_empty_depth{8};
};

struct RunExperimentConfig {
    HeuristicChoice heuristic{HeuristicChoice::h1};
    std::vector<NamedParameter> weight_overrides;

    SearchMode search{SearchMode::fixed};
    std::uint32_t fixed_depth{4};
    double time_limit_ms{0.0};
    bool use_adaptive_schedule{false};
    AdaptiveDepths adaptive_depths{};

    bool transposition_table{true};
    std::size_t transposition_table_capacity{1U << 16U};
    double probability_cutoff{0.0};
    bool symmetry{false};

    std::string seed_set_label;
    RunConfig run_config{};

    std::filesystem::path output_dir;
    // Weight file for learned evaluators (N-series). Required for N1,
    // rejected for the hand-crafted H-series.
    std::filesystem::path weights_path;
    bool quiet{false};
};

// Parses arguments given as plain strings (not raw argv) so this is directly
// unit-testable without spawning a process. Throws std::invalid_argument
// with a descriptive message on any invalid or contradictory combination.
[[nodiscard]] RunExperimentConfig parse_run_experiment_args(
    const std::vector<std::string>& args);

// Resolves a --seeds value ("quick", "standard", "final", or "FIRST-LAST")
// into a RunConfig plus a human-readable label for result metadata.
struct ResolvedSeeds {
    RunConfig run_config;
    std::string label;
};
[[nodiscard]] ResolvedSeeds resolve_seed_spec(
    const std::string& spec, std::optional<std::size_t> games_override);

}  // namespace adversarial_2048
