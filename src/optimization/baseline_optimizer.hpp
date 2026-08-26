#pragma once

#include "evaluation/baseline_heuristic.hpp"
#include "experiments/game_runner.hpp"
#include "search/expectimax.hpp"

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <string>
#include <vector>

namespace adversarial_2048 {

struct OptimizationStage {
    std::size_t game_count{};
    std::size_t survivor_count{};
};

struct BaselineOptimizationConfig {
    std::size_t candidate_count{16};
    std::uint32_t search_depth{3};
    std::uint64_t first_training_seed{1'000};
    std::uint64_t optimizer_seed{0x2048'BACEULL};
    double minimum_path_probability{0.0};
    double minimum_weight_multiplier{0.25};
    double maximum_weight_multiplier{4.0};
    std::string build_mode{"unknown"};
    std::string git_commit{"unavailable"};
    std::vector<OptimizationStage> stages{{3, 6}, {8, 3}, {20, 1}};
};

struct CandidateEvaluation {
    std::size_t stage_index{};
    std::size_t candidate_id{};
    BaselineWeights weights;
    ExperimentMetrics metrics;
    SearchStatistics search_statistics;
    double runtime_seconds{};
};

struct BaselineOptimizationResult {
    BaselineOptimizationConfig config;
    BaselineWeights best_weights;
    std::size_t best_candidate_id{};
    std::vector<CandidateEvaluation> evaluations;
    double runtime_seconds{};
};

using OptimizationProgress = std::function<void(const CandidateEvaluation&)>;

[[nodiscard]] BaselineOptimizationResult optimize_baseline_weights(
    const BaselineOptimizationConfig& config,
    const OptimizationProgress& progress = {});

struct OptimizationFiles {
    std::filesystem::path csv;
    std::filesystem::path json;
};

[[nodiscard]] OptimizationFiles write_optimization_results(
    const BaselineOptimizationResult& result,
    const std::filesystem::path& output_directory = "results");

}  // namespace adversarial_2048
