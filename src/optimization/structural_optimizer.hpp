#pragma once

#include "evaluation/structural_heuristic.hpp"
#include "experiments/game_runner.hpp"
#include "search/expectimax.hpp"

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <string>
#include <vector>

namespace adversarial_2048 {

struct StructuralOptimizationStage {
    std::size_t game_count{};
    std::size_t survivor_count{};
};

struct StructuralOptimizationConfig {
    std::size_t candidate_count{24};
    std::uint32_t search_depth{3};
    std::uint64_t first_training_seed{1'200};
    std::uint64_t optimizer_seed{0x51A7'2048ULL};
    double minimum_path_probability{0.0};
    bool optimize_baseline_weights{false};
    double minimum_baseline_weight_multiplier{0.5};
    double maximum_baseline_weight_multiplier{2.0};
    double minimum_weight_multiplier{0.125};
    double maximum_weight_multiplier{8.0};
    std::string build_mode{"unknown"};
    std::string git_commit{"unavailable"};
    std::vector<StructuralOptimizationStage> stages{{3, 8}, {8, 3}, {20, 1}};
};

struct StructuralCandidateEvaluation {
    std::size_t stage_index{};
    std::size_t candidate_id{};
    StructuralWeights weights;
    ExperimentMetrics metrics;
    SearchStatistics search_statistics;
    double runtime_seconds{};
};

struct StructuralOptimizationResult {
    StructuralOptimizationConfig config;
    StructuralWeights best_weights;
    std::size_t best_candidate_id{};
    std::vector<StructuralCandidateEvaluation> evaluations;
    double runtime_seconds{};
};

using StructuralOptimizationProgress =
    std::function<void(const StructuralCandidateEvaluation&)>;

[[nodiscard]] StructuralOptimizationResult optimize_structural_weights(
    const StructuralOptimizationConfig& config,
    const StructuralOptimizationProgress& progress = {});

struct StructuralOptimizationFiles {
    std::filesystem::path csv;
    std::filesystem::path json;
};

[[nodiscard]] StructuralOptimizationFiles write_structural_optimization_results(
    const StructuralOptimizationResult& result,
    const std::filesystem::path& output_directory = "results");

}  // namespace adversarial_2048
