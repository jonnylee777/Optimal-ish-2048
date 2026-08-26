#include "optimization/baseline_optimizer.hpp"

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
    BaselineWeights weights;
};

[[nodiscard]] double random_multiplier(
    RandomEngine& rng,
    double minimum,
    double maximum) {
    constexpr std::uint64_t resolution = 1'000'000;
    const auto unit = static_cast<double>(sample_bounded(rng, resolution + 1U)) /
                      static_cast<double>(resolution);
    return std::exp(std::log(minimum) + unit * (std::log(maximum) - std::log(minimum)));
}

void validate_config(const BaselineOptimizationConfig& config) {
    if (config.candidate_count == 0 || config.search_depth == 0 || config.stages.empty()) {
        throw std::invalid_argument("optimizer requires candidates, depth, and stages");
    }
    if (!(config.minimum_weight_multiplier > 0.0) ||
        !(config.maximum_weight_multiplier >= config.minimum_weight_multiplier)) {
        throw std::invalid_argument("invalid optimizer weight multiplier range");
    }
    if (!std::isfinite(config.minimum_path_probability) ||
        config.minimum_path_probability < 0.0 ||
        config.minimum_path_probability >= 1.0) {
        throw std::invalid_argument(
            "minimum path probability must be finite and in [0, 1)");
    }

    auto available = config.candidate_count;
    std::size_t previous_games = 0;
    for (const auto& stage : config.stages) {
        if (stage.game_count == 0 || stage.game_count < previous_games ||
            stage.survivor_count == 0 || stage.survivor_count > available) {
            throw std::invalid_argument("invalid optimizer stage sequence");
        }
        if (config.first_training_seed < seed_sets::training.first_seed ||
            config.first_training_seed + stage.game_count - 1U >=
                seed_sets::training.first_seed + seed_sets::training.game_count) {
            throw std::invalid_argument("optimizer games must remain inside training seeds");
        }
        previous_games = stage.game_count;
        available = stage.survivor_count;
    }
    if (available != 1) {
        throw std::invalid_argument("final optimizer stage must select one candidate");
    }
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

BaselineOptimizationResult optimize_baseline_weights(
    const BaselineOptimizationConfig& config,
    const OptimizationProgress& progress) {
    validate_config(config);
    const auto optimization_start = std::chrono::steady_clock::now();
    RandomEngine rng(config.optimizer_seed);
    const BaselineWeights initial{};
    std::vector<Candidate> candidates;
    candidates.reserve(config.candidate_count);
    candidates.push_back({0, initial});
    for (std::size_t id = 1; id < config.candidate_count; ++id) {
        candidates.push_back({
            id,
            {
                initial.empty_cells * random_multiplier(
                    rng, config.minimum_weight_multiplier, config.maximum_weight_multiplier),
                initial.monotonicity * random_multiplier(
                    rng, config.minimum_weight_multiplier, config.maximum_weight_multiplier),
                initial.smoothness * random_multiplier(
                    rng, config.minimum_weight_multiplier, config.maximum_weight_multiplier),
                initial.corner_preference * random_multiplier(
                    rng, config.minimum_weight_multiplier, config.maximum_weight_multiplier),
            },
        });
    }

    BaselineOptimizationResult result{config, {}, 0, {}, 0.0};
    for (std::size_t stage_index = 0; stage_index < config.stages.size(); ++stage_index) {
        const auto stage = config.stages[stage_index];
        std::vector<CandidateEvaluation> stage_results;
        stage_results.reserve(candidates.size());
        for (const auto& candidate : candidates) {
            const BaselineHeuristic evaluator(candidate.weights);
            SearchAgent agent(
                evaluator,
                config.search_depth,
                "baseline-candidate",
                {true, 1U << 16U, config.minimum_path_probability});
            const auto candidate_start = std::chrono::steady_clock::now();
            const auto experiment = GameRunner::run(
                agent,
                {stage.game_count, config.first_training_seed});
            CandidateEvaluation evaluation{
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
                  [](const CandidateEvaluation& left, const CandidateEvaluation& right) {
                      if (left.metrics.mean_score != right.metrics.mean_score) {
                          return left.metrics.mean_score > right.metrics.mean_score;
                      }
                      return left.candidate_id < right.candidate_id;
                  });
        candidates.clear();
        for (std::size_t index = 0; index < stage.survivor_count; ++index) {
            candidates.push_back({stage_results[index].candidate_id,
                                  stage_results[index].weights});
        }
    }

    result.best_candidate_id = candidates.front().id;
    result.best_weights = candidates.front().weights;
    result.runtime_seconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - optimization_start).count();
    return result;
}

OptimizationFiles write_optimization_results(
    const BaselineOptimizationResult& result,
    const std::filesystem::path& output_directory) {
    std::filesystem::create_directories(output_directory);
    const auto stem = "baseline_optimization_depth" +
                      std::to_string(result.config.search_depth) + '_' +
                      timestamp_for_filename();
    const OptimizationFiles files{
        output_directory / (stem + ".csv"),
        output_directory / (stem + ".json"),
    };

    std::ofstream csv(files.csv);
    if (!csv) {
        throw std::runtime_error("failed to open optimization CSV");
    }
    csv << "build_mode,search_depth,minimum_path_probability,stage,candidate,games,"
           "mean_score,median_score,score_stddev,runtime_seconds,"
           "empty_cells,monotonicity,smoothness,corner_preference,search_nodes,"
           "game_ms_per_move,cache_hit_rate\n"
        << std::setprecision(17);
    for (const auto& evaluation : result.evaluations) {
        csv << result.config.build_mode << ',' << result.config.search_depth << ','
            << result.config.minimum_path_probability << ','
            << evaluation.stage_index << ','
            << evaluation.candidate_id << ','
            << evaluation.metrics.game_count << ',' << evaluation.metrics.mean_score << ','
            << evaluation.metrics.median_score << ','
            << evaluation.metrics.score_standard_deviation << ','
            << evaluation.runtime_seconds << ','
            << evaluation.weights.empty_cells << ','
            << evaluation.weights.monotonicity << ','
            << evaluation.weights.smoothness << ','
            << evaluation.weights.corner_preference << ','
            << evaluation.search_statistics.total_nodes() << ','
            << evaluation.metrics.mean_milliseconds_per_move << ','
            << evaluation.search_statistics.cache_hit_rate() << '\n';
    }
    csv.close();
    if (!csv) {
        throw std::runtime_error("failed to write optimization CSV");
    }

    std::ofstream json(files.json);
    if (!json) {
        throw std::runtime_error("failed to open optimization JSON");
    }
    json << std::setprecision(17)
         << "{\n"
         << "  \"schema_version\": 1,\n"
         << "  \"optimizer\": \"deterministic-multistage-random-search\",\n"
         << "  \"objective\": \"mean game score\",\n"
         << "  \"build_mode\": \"" << result.config.build_mode << "\",\n"
         << "  \"git_commit\": \"" << result.config.git_commit << "\",\n"
         << "  \"search_depth\": " << result.config.search_depth << ",\n"
         << "  \"minimum_path_probability\": "
         << result.config.minimum_path_probability << ",\n"
         << "  \"first_training_seed\": " << result.config.first_training_seed << ",\n"
         << "  \"optimizer_seed\": " << result.config.optimizer_seed << ",\n"
         << "  \"candidate_count\": " << result.config.candidate_count << ",\n"
         << "  \"minimum_weight_multiplier\": "
         << result.config.minimum_weight_multiplier << ",\n"
         << "  \"maximum_weight_multiplier\": "
         << result.config.maximum_weight_multiplier << ",\n"
         << "  \"runtime_seconds\": " << result.runtime_seconds << ",\n"
         << "  \"best_candidate_id\": " << result.best_candidate_id << ",\n"
         << "  \"best_weights\": {\n"
         << "    \"empty_cells\": " << result.best_weights.empty_cells << ",\n"
         << "    \"monotonicity\": " << result.best_weights.monotonicity << ",\n"
         << "    \"smoothness\": " << result.best_weights.smoothness << ",\n"
         << "    \"corner_preference\": "
         << result.best_weights.corner_preference << "\n"
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
        throw std::runtime_error("failed to write optimization JSON");
    }
    return files;
}

}  // namespace adversarial_2048
