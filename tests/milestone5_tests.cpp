#include "optimization/baseline_optimizer.hpp"

#include <chrono>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace a2048 = adversarial_2048;

namespace {

class TestFailure : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

#define CHECK(condition)                                                                          \
    do {                                                                                          \
        if (!(condition)) {                                                                       \
            throw TestFailure(std::string("check failed: ") + #condition + " at line " +         \
                              std::to_string(__LINE__));                                           \
        }                                                                                         \
    } while (false)

[[nodiscard]] a2048::BaselineOptimizationConfig small_config() {
    a2048::BaselineOptimizationConfig config;
    config.candidate_count = 6;
    config.search_depth = 1;
    config.optimizer_seed = 123;
    config.stages = {{1, 1}};
    return config;
}

void test_optimizer_is_deterministic() {
    const auto first = a2048::optimize_baseline_weights(small_config());
    const auto second = a2048::optimize_baseline_weights(small_config());
    CHECK(first.best_candidate_id == second.best_candidate_id);
    CHECK(first.best_weights.empty_cells == second.best_weights.empty_cells);
    CHECK(first.best_weights.monotonicity == second.best_weights.monotonicity);
    CHECK(first.best_weights.smoothness == second.best_weights.smoothness);
    CHECK(first.best_weights.corner_preference == second.best_weights.corner_preference);
    CHECK(first.evaluations.size() == 6);
    CHECK(first.evaluations.size() == second.evaluations.size());
    for (std::size_t index = 0; index < first.evaluations.size(); ++index) {
        CHECK(first.evaluations[index].candidate_id == second.evaluations[index].candidate_id);
        CHECK(first.evaluations[index].metrics.mean_score ==
              second.evaluations[index].metrics.mean_score);
    }
}

void test_human_baseline_is_preserved_as_candidate_zero() {
    const auto result = a2048::optimize_baseline_weights(small_config());
    const auto& first = result.evaluations.front();
    const a2048::BaselineWeights initial;
    CHECK(first.candidate_id == 0);
    CHECK(first.weights.empty_cells == initial.empty_cells);
    CHECK(first.weights.monotonicity == initial.monotonicity);
    CHECK(first.weights.smoothness == initial.smoothness);
    CHECK(first.weights.corner_preference == initial.corner_preference);
}

void test_frozen_optimized_weights() {
    const auto weights = a2048::kDepth3OptimizedBaselineWeights;
    CHECK(weights.empty_cells == 85.221621525239641);
    CHECK(weights.monotonicity == 44.83062947862652);
    CHECK(weights.smoothness == 43.676658929747916);
    CHECK(weights.corner_preference == 28.751796341054963);
}

void test_invalid_configuration_is_rejected() {
    auto config = small_config();
    config.search_depth = 0;
    bool rejected = false;
    try {
        static_cast<void>(a2048::optimize_baseline_weights(config));
    } catch (const std::invalid_argument&) {
        rejected = true;
    }
    CHECK(rejected);

    config = small_config();
    config.first_training_seed = 10'000;
    rejected = false;
    try {
        static_cast<void>(a2048::optimize_baseline_weights(config));
    } catch (const std::invalid_argument&) {
        rejected = true;
    }
    CHECK(rejected);

    config = small_config();
    config.minimum_path_probability = 1.0;
    rejected = false;
    try {
        static_cast<void>(a2048::optimize_baseline_weights(config));
    } catch (const std::invalid_argument&) {
        rejected = true;
    }
    CHECK(rejected);
}

void test_optimization_result_files() {
    const auto result = a2048::optimize_baseline_weights(small_config());
    const auto unique = std::chrono::steady_clock::now().time_since_epoch().count();
    const auto output_directory = std::filesystem::temp_directory_path() /
                                  ("adversarial_2048_optimizer_test_" +
                                   std::to_string(unique));
    const auto files = a2048::write_optimization_results(result, output_directory);
    CHECK(std::filesystem::is_regular_file(files.csv));
    CHECK(std::filesystem::is_regular_file(files.json));

    std::ifstream json_stream(files.json);
    const std::string json_text{
        std::istreambuf_iterator<char>{json_stream}, std::istreambuf_iterator<char>{}};
    CHECK(json_text.find("deterministic-multistage-random-search") != std::string::npos);
    CHECK(json_text.find("\"best_weights\"") != std::string::npos);
    CHECK(json_text.find("\"minimum_path_probability\": 0") != std::string::npos);

    std::ifstream csv_stream(files.csv);
    std::string csv_header;
    std::getline(csv_stream, csv_header);
    CHECK(csv_header.find("search_depth") != std::string::npos);
    CHECK(csv_header.find("minimum_path_probability") != std::string::npos);

    std::filesystem::remove_all(output_directory);
}

}  // namespace

int main() {
    const std::vector<std::pair<std::string, void (*)()>> tests{
        {"optimizer determinism", test_optimizer_is_deterministic},
        {"baseline candidate", test_human_baseline_is_preserved_as_candidate_zero},
        {"frozen optimized weights", test_frozen_optimized_weights},
        {"invalid config", test_invalid_configuration_is_rejected},
        {"optimization files", test_optimization_result_files},
    };

    std::size_t failures = 0;
    for (const auto& [name, test] : tests) {
        try {
            test();
            std::cout << "[PASS] " << name << '\n';
        } catch (const std::exception& error) {
            ++failures;
            std::cerr << "[FAIL] " << name << ": " << error.what() << '\n';
        }
    }

    std::cout << tests.size() - failures << '/' << tests.size() << " tests passed\n";
    return failures == 0 ? 0 : 1;
}
