#pragma once

#include "experiments/game_runner.hpp"
#include "search/expectimax.hpp"

#include <cstdint>
#include <array>
#include <filesystem>
#include <optional>
#include <string>
#include <vector>

namespace adversarial_2048 {

struct NamedParameter {
    std::string name;
    double value{};
};

struct ResultMetadata {
    std::string evaluator_type;
    std::vector<NamedParameter> evaluator_parameters;
    std::string feature_configuration;
    std::uint32_t search_depth{};
    std::string depth_definition;
    std::string seed_partition;
    std::string build_mode;
    std::string git_commit{"unavailable"};
    std::string optimization_configuration{"none"};
    std::optional<SearchStatistics> search_statistics;
    double minimum_path_probability{};
    std::optional<std::array<std::uint64_t, 3>> adaptive_depth_usage;
    double time_limit_seconds{};
    std::optional<std::array<std::uint64_t, 13>> completed_depth_usage;
    // Fraction of moves whose completed_depth fell short of the requested
    // ceiling. Only computable when every move shares one requested ceiling
    // (fixed-depth runs, or timed runs with a flat depth ceiling); left unset
    // for the adaptive empty-cell schedule, where completed_depth_usage and
    // adaptive_depth_usage are independent aggregates with no per-move link
    // recorded, so an exact per-move hit rate isn't reconstructable yet (see
    // ROADMAP.md #1, per-move instrumentation).
    std::optional<double> deadline_hit_rate;
};

struct ResultFiles {
    std::filesystem::path csv;
    std::filesystem::path json;
};

[[nodiscard]] ResultFiles write_experiment_results(
    const ExperimentResult& experiment,
    const ResultMetadata& metadata,
    const std::filesystem::path& output_directory = "results");

}  // namespace adversarial_2048
