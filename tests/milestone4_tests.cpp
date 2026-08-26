#include "experiments/game_runner.hpp"
#include "experiments/result_writer.hpp"
#include "experiments/seed_sets.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
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

[[nodiscard]] bool approximately_equal(double left, double right, double tolerance = 1e-9) {
    return std::abs(left - right) <= tolerance;
}

[[nodiscard]] a2048::ExperimentResult fixture_result() {
    return {
        "baseline",
        {4, 10'000},
        {
            {10'000, 100, 10, 10, 0.1},
            {10'001, 200, 20, 11, 0.2},
            {10'002, 300, 30, 12, 0.3},
            {10'003, 400, 40, 13, 0.4},
        },
        1.0,
    };
}

void test_seed_partitions() {
    CHECK(a2048::seed_sets::classify(1'000, 1'999) == "training");
    CHECK(a2048::seed_sets::classify(10'000, 10'499) == "validation");
    CHECK(a2048::seed_sets::classify(50'000, 50'999) == "final-test");
    CHECK(a2048::seed_sets::classify(9'999, 10'000) == "custom");
}

void test_summary_statistics() {
    const auto metrics = fixture_result().metrics();
    CHECK(metrics.game_count == 4);
    CHECK(metrics.mean_score == 250.0);
    CHECK(metrics.median_score == 250.0);
    CHECK(approximately_equal(metrics.score_standard_deviation, std::sqrt(50'000.0 / 3.0)));
    const auto expected_margin = 1.96 * metrics.score_standard_deviation / 2.0;
    CHECK(approximately_equal(metrics.score_confidence_95_low, 250.0 - expected_margin));
    CHECK(approximately_equal(metrics.score_confidence_95_high, 250.0 + expected_margin));
    CHECK(metrics.best_score == 400);
    CHECK(metrics.mean_max_tile == 3'840.0);
    CHECK(metrics.highest_tile == 8'192);
    CHECK(metrics.mode_max_tile == 8'192);
    CHECK(metrics.achievement_rates.tile_1024 == 1.0);
    CHECK(metrics.achievement_rates.tile_2048 == 0.75);
    CHECK(metrics.achievement_rates.tile_4096 == 0.5);
    CHECK(metrics.achievement_rates.tile_8192 == 0.25);
    CHECK(metrics.mean_moves == 25.0);
    CHECK(approximately_equal(metrics.mean_runtime_seconds, 0.25));
    CHECK(approximately_equal(metrics.mean_milliseconds_per_move, 10.0));
}

void test_result_files() {
    const auto unique = std::chrono::steady_clock::now().time_since_epoch().count();
    const auto output_directory = std::filesystem::temp_directory_path() /
                                  ("adversarial_2048_results_test_" +
                                   std::to_string(unique));
    const a2048::ResultMetadata metadata{
        "baseline-heuristic",
        {{"empty_cells", 270.0}, {"monotonicity", 47.0}},
        "baseline-v1:test",
        2,
        "player layers",
        "validation",
        "Test",
        "unavailable",
        "none",
        a2048::SearchStatistics{1, 2, 3, 4, 5, 2, 0.01},
        0.0,
        std::nullopt,
        0.0,
        std::nullopt,
    };

    const auto files = a2048::write_experiment_results(
        fixture_result(), metadata, output_directory);
    CHECK(std::filesystem::is_regular_file(files.csv));
    CHECK(std::filesystem::is_regular_file(files.json));

    std::ifstream csv_stream(files.csv);
    const std::string csv_text{
        std::istreambuf_iterator<char>{csv_stream}, std::istreambuf_iterator<char>{}};
    CHECK(csv_text.find("evaluator_parameters") != std::string::npos);
    CHECK(csv_text.find("empty_cells=270") != std::string::npos);
    CHECK(std::count(csv_text.begin(), csv_text.end(), '\n') == 5);

    std::ifstream json_stream(files.json);
    const std::string json_text{
        std::istreambuf_iterator<char>{json_stream}, std::istreambuf_iterator<char>{}};
    CHECK(json_text.find("\"schema_version\": 1") != std::string::npos);
    CHECK(json_text.find("\"mean_score\": 250") != std::string::npos);
    CHECK(json_text.find("\"mode_max_tile\": 8192") != std::string::npos);
    CHECK(json_text.find("\"seed_partition\": \"validation\"") != std::string::npos);
    CHECK(json_text.find("\"empty_cells\": 270") != std::string::npos);

    std::filesystem::remove_all(output_directory);
}

void test_empty_experiment_is_rejected() {
    bool rejected = false;
    try {
        static_cast<void>(a2048::write_experiment_results(
            a2048::ExperimentResult{}, a2048::ResultMetadata{}));
    } catch (const std::invalid_argument&) {
        rejected = true;
    }
    CHECK(rejected);
}

}  // namespace

int main() {
    const std::vector<std::pair<std::string, void (*)()>> tests{
        {"seed partitions", test_seed_partitions},
        {"summary statistics", test_summary_statistics},
        {"result files", test_result_files},
        {"empty experiment", test_empty_experiment_is_rejected},
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
