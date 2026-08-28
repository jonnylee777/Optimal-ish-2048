#include "experiments/game_runner.hpp"
#include "experiments/run_experiment_config.hpp"
#include "experiments/seed_sets.hpp"

#include <cstddef>
#include <exception>
#include <filesystem>
#include <iostream>
#include <optional>
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

#define CHECK_THROWS(expression)                                                                 \
    do {                                                                                          \
        bool threw = false;                                                                       \
        try {                                                                                     \
            (void)(expression);                                                                    \
        } catch (const std::exception&) {                                                          \
            threw = true;                                                                          \
        }                                                                                          \
        if (!threw) {                                                                              \
            throw TestFailure(std::string("expected throw: ") + #expression + " at line " +       \
                              std::to_string(__LINE__));                                           \
        }                                                                                          \
    } while (false)

void test_seed_spec_named_sets() {
    const auto quick = a2048::resolve_seed_spec("quick", std::nullopt);
    CHECK(quick.run_config.game_count == a2048::seed_sets::quick_benchmark.game_count);
    CHECK(quick.run_config.first_seed == a2048::seed_sets::quick_benchmark.first_seed);
    CHECK(quick.label == "quick-benchmark");

    const auto standard = a2048::resolve_seed_spec("standard", std::nullopt);
    CHECK(standard.run_config.game_count == a2048::seed_sets::standard_benchmark.game_count);
    CHECK(standard.label == "standard-benchmark");

    const auto final_set = a2048::resolve_seed_spec("final", std::nullopt);
    CHECK(final_set.run_config.game_count == a2048::seed_sets::final_benchmark.game_count);
    CHECK(final_set.label == "final-benchmark");
}

void test_seed_spec_custom_range() {
    const auto custom = a2048::resolve_seed_spec("500-509", std::nullopt);
    CHECK(custom.run_config.first_seed == 500);
    CHECK(custom.run_config.game_count == 10);
    CHECK(custom.label == "custom");

    CHECK_THROWS(a2048::resolve_seed_spec("509-500", std::nullopt));
    CHECK_THROWS(a2048::resolve_seed_spec("not-a-range", std::nullopt));
}

void test_seed_spec_games_override() {
    const auto reduced = a2048::resolve_seed_spec("quick", std::size_t{10});
    CHECK(reduced.run_config.game_count == 10);
    CHECK(reduced.run_config.first_seed == a2048::seed_sets::quick_benchmark.first_seed);

    CHECK_THROWS(a2048::resolve_seed_spec(
        "quick", a2048::seed_sets::quick_benchmark.game_count + 1U));
    CHECK_THROWS(a2048::resolve_seed_spec("quick", std::size_t{0}));
    CHECK_THROWS(a2048::resolve_seed_spec("500-509", std::size_t{5}));
}

void test_fixed_search_requires_depth() {
    CHECK_THROWS(a2048::parse_run_experiment_args(
        {"--heuristic", "H1", "--search", "fixed", "--seeds", "quick"}));
}

void test_fixed_search_rejects_time_limit() {
    CHECK_THROWS(a2048::parse_run_experiment_args(
        {"--heuristic", "H1", "--search", "fixed", "--depth", "4",
         "--time-limit-ms", "250", "--seeds", "quick"}));
}

void test_timed_search_requires_time_limit() {
    CHECK_THROWS(a2048::parse_run_experiment_args(
        {"--heuristic", "H1", "--search", "timed", "--seeds", "quick"}));
}

void test_timed_search_rejects_depth_and_schedule_together() {
    CHECK_THROWS(a2048::parse_run_experiment_args(
        {"--heuristic", "H1", "--search", "timed", "--time-limit-ms", "250",
         "--depth", "4", "--adaptive-schedule", "4,6,8", "--seeds", "quick"}));
}

void test_timed_search_defaults_to_adaptive_schedule() {
    const auto config = a2048::parse_run_experiment_args(
        {"--heuristic", "H1", "--search", "timed", "--time-limit-ms", "250", "--seeds", "quick"});
    CHECK(config.use_adaptive_schedule);
    CHECK(config.adaptive_depths.high_empty_depth == 4);
    CHECK(config.adaptive_depths.medium_empty_depth == 6);
    CHECK(config.adaptive_depths.low_empty_depth == 8);
}

void test_timed_search_with_flat_depth_ceiling() {
    const auto config = a2048::parse_run_experiment_args(
        {"--heuristic", "H0", "--search", "timed", "--time-limit-ms", "100",
         "--depth", "6", "--seeds", "quick"});
    CHECK(!config.use_adaptive_schedule);
    CHECK(config.fixed_depth == 6);
}

void test_unrecognized_flag_rejected() {
    CHECK_THROWS(a2048::parse_run_experiment_args({"--bogus", "value"}));
}

void test_full_valid_fixed_config() {
    const auto config = a2048::parse_run_experiment_args(
        {"--heuristic", "H0", "--weight", "empty_cells=5", "--search", "fixed",
         "--depth", "4", "--transposition-table", "off", "--seeds", "quick", "--games", "10",
         "--quiet"});
    CHECK(config.heuristic == a2048::HeuristicChoice::h0);
    CHECK(config.weight_overrides.size() == 1);
    CHECK(config.weight_overrides[0].name == "empty_cells");
    CHECK(config.weight_overrides[0].value == 5.0);
    CHECK(config.search == a2048::SearchMode::fixed);
    CHECK(config.fixed_depth == 4);
    CHECK(config.time_limit_ms == 0.0);
    CHECK(!config.transposition_table);
    CHECK(config.run_config.game_count == 10);
    CHECK(config.quiet);
    // Results are filed by METHODOLOGY (hand-written vs learned), not by search
    // regime. This config uses H0, so it lands in the heuristics directory
    // whether it runs fixed-depth or timed -- comparing the two regimes for one
    // agent is the point, so they belong together.
    CHECK(config.output_dir == std::filesystem::path("experiments/results/phase1-heuristics"));
}

void test_metrics_worst_score_median_moves_and_distribution() {
    a2048::ExperimentResult result;
    result.agent_name = "test";
    result.games = {
        a2048::GameRecord{1, 100, 10, 5, 1.0},
        a2048::GameRecord{2, 300, 20, 10, 2.0},
        a2048::GameRecord{3, 200, 15, 14, 1.5},
        a2048::GameRecord{4, 400, 30, 16, 3.0},
    };
    const auto metrics = result.metrics();
    CHECK(metrics.worst_score == 100);
    CHECK(metrics.best_score == 400);
    CHECK(metrics.median_moves == 17.5);
    CHECK(metrics.max_tile_distribution[5] == 1);
    CHECK(metrics.max_tile_distribution[10] == 1);
    CHECK(metrics.max_tile_distribution[14] == 1);
    CHECK(metrics.max_tile_distribution[16] == 1);
    CHECK(metrics.achievement_rates.tile_16384 == 0.5);
    CHECK(metrics.achievement_rates.tile_32768 == 0.25);
    CHECK(metrics.achievement_rates.tile_65536 == 0.25);
}


// The depth schedule must be usable WITHOUT a time limit. Coupling the two made
// the schedule untestable: a timed run's strength depends on machine speed and
// on how many games share the machine, so it can be neither reproduced nor
// parallelised. Fixed + schedule is deterministic and therefore benchmarkable.
void test_fixed_search_accepts_adaptive_schedule() {
    const auto config = a2048::parse_run_experiment_args(
        {"--heuristic", "N1", "--weights", "w.bin", "--search", "fixed",
         "--adaptive-schedule", "4,6,8", "--seeds", "30000-30009"});
    CHECK(config.use_adaptive_schedule);
    CHECK(config.adaptive_depths.high_empty_depth == 4);
    CHECK(config.adaptive_depths.medium_empty_depth == 6);
    CHECK(config.adaptive_depths.low_empty_depth == 8);
    CHECK(config.time_limit_ms == 0.0);

    // Still mutually exclusive with an explicit depth, and one of them is still
    // required -- lifting the restriction must not make the depth optional.
    CHECK_THROWS(a2048::parse_run_experiment_args(
        {"--heuristic", "H1", "--search", "fixed", "--depth", "4",
         "--adaptive-schedule", "4,6,8", "--seeds", "quick"}));
    CHECK_THROWS(a2048::parse_run_experiment_args(
        {"--heuristic", "H1", "--search", "fixed", "--seeds", "quick"}));
    CHECK_THROWS(a2048::parse_run_experiment_args(
        {"--heuristic", "H1", "--search", "fixed", "--adaptive-schedule", "4,0,8",
         "--seeds", "quick"}));
    CHECK_THROWS(a2048::parse_run_experiment_args(
        {"--heuristic", "H1", "--search", "fixed", "--adaptive-schedule", "4,6,8",
         "--time-limit-ms", "250", "--seeds", "quick"}));
}

}  // namespace

int main() {
    const std::vector<std::pair<std::string, void (*)()>> tests{
        {"seed spec named sets", test_seed_spec_named_sets},
        {"seed spec custom range", test_seed_spec_custom_range},
        {"seed spec games override", test_seed_spec_games_override},
        {"fixed search requires depth", test_fixed_search_requires_depth},
        {"fixed search rejects time limit", test_fixed_search_rejects_time_limit},
        {"timed search requires time limit", test_timed_search_requires_time_limit},
        {"timed search rejects depth+schedule", test_timed_search_rejects_depth_and_schedule_together},
        {"timed search defaults to adaptive schedule", test_timed_search_defaults_to_adaptive_schedule},
        {"timed search with flat depth ceiling", test_timed_search_with_flat_depth_ceiling},
        {"unrecognized flag rejected", test_unrecognized_flag_rejected},
        {"fixed search accepts an adaptive schedule",
         test_fixed_search_accepts_adaptive_schedule},
        {"full valid fixed config", test_full_valid_fixed_config},
        {"metrics worst/median/distribution", test_metrics_worst_score_median_moves_and_distribution},
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
