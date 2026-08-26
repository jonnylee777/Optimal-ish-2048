#include "optimization/structural_optimizer.hpp"

#include "agents/search_agent.hpp"
#include "core/random.hpp"
#include "experiments/seed_sets.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <ctime>
#include <fstream>
#include <iomanip>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>

namespace adversarial_2048 {
namespace {

struct Candidate {
    std::size_t id{};
    StructuralWeights weights;
};

[[nodiscard]] StructuralWeights zero_structural_weights() noexcept {
    StructuralWeights weights{};
    weights.main_line = 0.0;
    weights.structural_stability = 0.0;
    weights.adverse_stuck = 0.0;
    weights.structural_displacement = 0.0;
    return weights;
}

[[nodiscard]] double random_multiplier(
    RandomEngine& rng, double minimum, double maximum) {
    constexpr std::uint64_t resolution = 1'000'000;
    const auto unit = static_cast<double>(sample_bounded(rng, resolution + 1U)) /
                      static_cast<double>(resolution);
    return std::exp(std::log(minimum) + unit * (std::log(maximum) - std::log(minimum)));
}

void normalize_joint_weights(StructuralWeights& weights) {
    const StructuralWeights initial{};
    const auto total = [](const StructuralWeights& value) {
        return value.baseline.empty_cells + value.baseline.monotonicity +
               value.baseline.smoothness + value.baseline.corner_preference +
               value.main_line + value.structural_stability +
               value.adverse_stuck + value.structural_displacement;
    };
    const auto candidate_total = total(weights);
    const auto scale = total(initial) / candidate_total;
    weights.baseline.empty_cells *= scale;
    weights.baseline.monotonicity *= scale;
    weights.baseline.smoothness *= scale;
    weights.baseline.corner_preference *= scale;
    weights.main_line *= scale;
    weights.structural_stability *= scale;
    weights.adverse_stuck *= scale;
    weights.structural_displacement *= scale;
}

void validate_config(const StructuralOptimizationConfig& config) {
    if (config.candidate_count < 4 || config.search_depth == 0 || config.stages.empty()) {
        throw std::invalid_argument(
            "structural optimizer requires at least four candidates, depth, and stages");
    }
    if (!std::isfinite(config.minimum_path_probability) ||
        config.minimum_path_probability < 0.0 ||
        config.minimum_path_probability >= 1.0) {
        throw std::invalid_argument(
            "minimum path probability must be finite and in [0, 1)");
    }
    if (!(config.minimum_weight_multiplier > 0.0) ||
        !(config.maximum_weight_multiplier >= config.minimum_weight_multiplier)) {
        throw std::invalid_argument("invalid structural weight multiplier range");
    }
    if (!(config.minimum_baseline_weight_multiplier > 0.0) ||
        !(config.maximum_baseline_weight_multiplier >=
          config.minimum_baseline_weight_multiplier)) {
        throw std::invalid_argument("invalid baseline weight multiplier range");
    }

    auto available = config.candidate_count;
    std::size_t previous_games = 0;
    for (const auto& stage : config.stages) {
        if (stage.game_count == 0 || stage.game_count < previous_games ||
            stage.survivor_count == 0 || stage.survivor_count > available) {
            throw std::invalid_argument("invalid structural optimizer stage sequence");
        }
        const auto final_seed = config.first_training_seed + stage.game_count - 1U;
        if (config.first_training_seed < seed_sets::training.first_seed ||
            final_seed >= seed_sets::training.first_seed + seed_sets::training.game_count) {
            throw std::invalid_argument(
                "structural optimizer games must remain inside training seeds");
        }
        previous_games = stage.game_count;
        available = stage.survivor_count;
    }
    if (available != 1) {
        throw std::invalid_argument(
            "final structural optimizer stage must select one candidate");
    }
}

[[nodiscard]] std::vector<Candidate> make_candidates(
    const StructuralOptimizationConfig& config) {
    std::vector<Candidate> candidates;
    candidates.reserve(config.candidate_count);
    candidates.push_back({0, zero_structural_weights()});
    candidates.push_back({1, StructuralWeights{}});
    candidates.push_back({2, main_line_ablation_weights()});
    candidates.push_back({3, movement_ablation_weights()});

    RandomEngine rng(config.optimizer_seed);
    const StructuralWeights initial{};
    for (std::size_t id = candidates.size(); id < config.candidate_count; ++id) {
        auto weights = initial;
        if (config.optimize_baseline_weights) {
            weights.baseline.empty_cells *= random_multiplier(
                rng,
                config.minimum_baseline_weight_multiplier,
                config.maximum_baseline_weight_multiplier);
            weights.baseline.monotonicity *= random_multiplier(
                rng,
                config.minimum_baseline_weight_multiplier,
                config.maximum_baseline_weight_multiplier);
            weights.baseline.smoothness *= random_multiplier(
                rng,
                config.minimum_baseline_weight_multiplier,
                config.maximum_baseline_weight_multiplier);
            weights.baseline.corner_preference *= random_multiplier(
                rng,
                config.minimum_baseline_weight_multiplier,
                config.maximum_baseline_weight_multiplier);
        }
        weights.main_line *= random_multiplier(
            rng, config.minimum_weight_multiplier, config.maximum_weight_multiplier);
        weights.structural_stability *= random_multiplier(
            rng, config.minimum_weight_multiplier, config.maximum_weight_multiplier);
        weights.adverse_stuck *= random_multiplier(
            rng, config.minimum_weight_multiplier, config.maximum_weight_multiplier);
        weights.structural_displacement *= random_multiplier(
            rng, config.minimum_weight_multiplier, config.maximum_weight_multiplier);
        if (config.optimize_baseline_weights) {
            normalize_joint_weights(weights);
        }
        candidates.push_back({id, weights});
    }
    return candidates;
}

[[nodiscard]] std::string timestamp_for_filename() {
    const auto now = std::chrono::system_clock::now();
    const auto time = std::chrono::system_clock::to_time_t(now);
    std::tm utc{};
#if defined(_WIN32)
    gmtime_s(&utc, &time);
#else
    gmtime_r(&time, &utc);
#endif
    std::ostringstream text;
    text << std::put_time(&utc, "%Y%m%dT%H%M%SZ");
    return text.str();
}

}  // namespace

StructuralOptimizationResult optimize_structural_weights(
    const StructuralOptimizationConfig& config,
    const StructuralOptimizationProgress& progress) {
    validate_config(config);
    const auto optimization_start = std::chrono::steady_clock::now();
    auto candidates = make_candidates(config);
    StructuralOptimizationResult result{config, {}, 0, {}, 0.0};

    for (std::size_t stage_index = 0; stage_index < config.stages.size(); ++stage_index) {
        const auto stage = config.stages[stage_index];
        std::vector<StructuralCandidateEvaluation> stage_results;
        stage_results.reserve(candidates.size());
        for (const auto& candidate : candidates) {
            const StructuralHeuristic evaluator(candidate.weights);
            SearchAgent agent(
                evaluator,
                config.search_depth,
                "structural-candidate",
                {true, 1U << 16U, config.minimum_path_probability});
            const auto candidate_start = std::chrono::steady_clock::now();
            const auto experiment = GameRunner::run(
                agent, {stage.game_count, config.first_training_seed});
            StructuralCandidateEvaluation evaluation{
                stage_index,
                candidate.id,
                candidate.weights,
                experiment.metrics(),
                agent.cumulative_statistics(),
                std::chrono::duration<double>(
                    std::chrono::steady_clock::now() - candidate_start).count(),
            };
            stage_results.push_back(evaluation);
            result.evaluations.push_back(evaluation);
            if (progress) {
                progress(evaluation);
            }
        }

        std::sort(stage_results.begin(), stage_results.end(),
                  [](const auto& left, const auto& right) {
                      if (left.metrics.mean_score != right.metrics.mean_score) {
                          return left.metrics.mean_score > right.metrics.mean_score;
                      }
                      return left.candidate_id < right.candidate_id;
                  });
        candidates.clear();
        for (std::size_t index = 0; index < stage.survivor_count; ++index) {
            candidates.push_back({
                stage_results[index].candidate_id,
                stage_results[index].weights,
            });
        }
    }

    result.best_candidate_id = candidates.front().id;
    result.best_weights = candidates.front().weights;
    result.runtime_seconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - optimization_start).count();
    return result;
}

StructuralOptimizationFiles write_structural_optimization_results(
    const StructuralOptimizationResult& result,
    const std::filesystem::path& output_directory) {
    std::filesystem::create_directories(output_directory);
    const auto stem = "structural_optimization_depth" +
                      std::to_string(result.config.search_depth) + '_' +
                      timestamp_for_filename();
    const StructuralOptimizationFiles files{
        output_directory / (stem + ".csv"),
        output_directory / (stem + ".json"),
    };

    std::ofstream csv(files.csv);
    if (!csv) {
        throw std::runtime_error("failed to open structural optimization CSV");
    }
    csv << "build_mode,search_depth,minimum_path_probability,optimize_baseline_weights,"
           "stage,candidate,games,"
           "mean_score,median_score,score_stddev,runtime_seconds,main_line,"
           "structural_stability,adverse_stuck,structural_displacement,empty_cells,"
           "monotonicity,smoothness,corner_preference,search_nodes,"
           "game_ms_per_move,cache_hit_rate\n"
        << std::setprecision(17);
    for (const auto& evaluation : result.evaluations) {
        csv << result.config.build_mode << ',' << result.config.search_depth << ','
            << result.config.minimum_path_probability << ','
            << (result.config.optimize_baseline_weights ? "true" : "false") << ','
            << evaluation.stage_index + 1U << ',' << evaluation.candidate_id << ','
            << evaluation.metrics.game_count << ',' << evaluation.metrics.mean_score << ','
            << evaluation.metrics.median_score << ','
            << evaluation.metrics.score_standard_deviation << ','
            << evaluation.runtime_seconds << ',' << evaluation.weights.main_line << ','
            << evaluation.weights.structural_stability << ','
            << evaluation.weights.adverse_stuck << ','
            << evaluation.weights.structural_displacement << ','
            << evaluation.weights.baseline.empty_cells << ','
            << evaluation.weights.baseline.monotonicity << ','
            << evaluation.weights.baseline.smoothness << ','
            << evaluation.weights.baseline.corner_preference << ','
            << evaluation.search_statistics.total_nodes() << ','
            << evaluation.metrics.mean_milliseconds_per_move << ','
            << evaluation.search_statistics.cache_hit_rate() << '\n';
    }
    csv.close();
    if (!csv) {
        throw std::runtime_error("failed to write structural optimization CSV");
    }

    std::ofstream json(files.json);
    if (!json) {
        throw std::runtime_error("failed to open structural optimization JSON");
    }
    const auto& baseline = result.best_weights.baseline;
    json << std::setprecision(17)
         << "{\n"
         << "  \"schema_version\": 1,\n"
         << "  \"optimizer\": \"deterministic-multistage-structural-random-search\",\n"
         << "  \"objective\": \"mean game score\",\n"
         << "  \"build_mode\": \"" << result.config.build_mode << "\",\n"
         << "  \"git_commit\": \"" << result.config.git_commit << "\",\n"
         << "  \"search_depth\": " << result.config.search_depth << ",\n"
         << "  \"minimum_path_probability\": "
         << result.config.minimum_path_probability << ",\n"
         << "  \"optimize_baseline_weights\": "
         << (result.config.optimize_baseline_weights ? "true" : "false") << ",\n"
         << "  \"joint_weight_normalization\": \""
         << (result.config.optimize_baseline_weights
                 ? "sum-to-initial-weight-sum"
                 : "not-applicable")
         << "\",\n"
         << "  \"first_training_seed\": "
         << result.config.first_training_seed << ",\n"
         << "  \"optimizer_seed\": " << result.config.optimizer_seed << ",\n"
         << "  \"candidate_count\": " << result.config.candidate_count << ",\n"
         << "  \"minimum_weight_multiplier\": "
         << result.config.minimum_weight_multiplier << ",\n"
         << "  \"maximum_weight_multiplier\": "
         << result.config.maximum_weight_multiplier << ",\n"
         << "  \"minimum_baseline_weight_multiplier\": "
         << result.config.minimum_baseline_weight_multiplier << ",\n"
         << "  \"maximum_baseline_weight_multiplier\": "
         << result.config.maximum_baseline_weight_multiplier << ",\n"
         << "  \"runtime_seconds\": " << result.runtime_seconds << ",\n"
         << "  \"best_candidate_id\": " << result.best_candidate_id << ",\n"
         << "  \"best_baseline_weights\": {\n"
         << "    \"empty_cells\": " << baseline.empty_cells << ",\n"
         << "    \"monotonicity\": " << baseline.monotonicity << ",\n"
         << "    \"smoothness\": " << baseline.smoothness << ",\n"
         << "    \"corner_preference\": " << baseline.corner_preference << "\n"
         << "  },\n"
         << "  \"best_structural_weights\": {\n"
         << "    \"main_line\": " << result.best_weights.main_line << ",\n"
         << "    \"structural_stability\": "
         << result.best_weights.structural_stability << ",\n"
         << "    \"adverse_stuck\": " << result.best_weights.adverse_stuck << ",\n"
         << "    \"structural_displacement\": "
         << result.best_weights.structural_displacement << "\n"
         << "  },\n"
         << "  \"controls\": {\n"
         << "    \"candidate_0\": \"all structural weights zero\",\n"
         << "    \"candidate_1\": \"current full structural weights\",\n"
         << "    \"candidate_2\": \"main-line ablation\",\n"
         << "    \"candidate_3\": \"main-line plus movement ablation\"\n"
         << "  },\n"
         << "  \"stages\": [";
    for (std::size_t index = 0; index < result.config.stages.size(); ++index) {
        const auto stage = result.config.stages[index];
        json << (index == 0 ? "\n" : ",\n")
             << "    {\"games\": " << stage.game_count
             << ", \"survivors\": " << stage.survivor_count << '}';
    }
    json << "\n  ]\n}\n";
    json.close();
    if (!json) {
        throw std::runtime_error("failed to write structural optimization JSON");
    }
    return files;
}

}  // namespace adversarial_2048
