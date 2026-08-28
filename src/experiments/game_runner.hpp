#pragma once

#include "agents/agent.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace adversarial_2048 {

struct RunConfig {
    std::size_t game_count{100};
    std::uint64_t first_seed{1'000};
};

struct GameRecord {
    std::uint64_t seed{};
    std::uint64_t score{};
    std::uint64_t moves{};
    std::uint8_t max_tile_exponent{};
    double runtime_seconds{};
};

struct AchievementRates {
    double tile_1024{};
    double tile_2048{};
    double tile_4096{};
    double tile_8192{};
    double tile_16384{};
    double tile_32768{};
    double tile_65536{};
};

struct ExperimentMetrics {
    std::size_t game_count{};
    double mean_score{};
    double median_score{};
    double score_standard_deviation{};
    double score_confidence_95_low{};
    double score_confidence_95_high{};
    std::uint64_t best_score{};
    std::uint64_t worst_score{};
    double mean_max_tile{};
    std::uint64_t highest_tile{};
    std::uint64_t mode_max_tile{};
    AchievementRates achievement_rates;
    double mean_moves{};
    double median_moves{};
    double mean_runtime_seconds{};
    double mean_milliseconds_per_move{};
    // Indexed by max-tile exponent (0 = no tiles/empty board never occurs at
    // game end, 1 = tile 2, ..., 31 = tile 2^31); value is the game count
    // that ended with that exponent as the board's maximum tile.
    std::array<std::uint64_t, 32> max_tile_distribution{};
};

struct ExperimentResult {
    std::string agent_name;
    RunConfig config;
    std::vector<GameRecord> games;
    double runtime_seconds{};
    // Worker threads that produced `games`. Scores are identical at any worker
    // count, but per-game `runtime_seconds` (and therefore every derived timing
    // metric) is contended above 1 and must not be quoted as a speed result.
    std::size_t worker_count{1};

    [[nodiscard]] double mean_score() const noexcept;
    [[nodiscard]] std::uint64_t best_score() const noexcept;
    [[nodiscard]] double mean_moves() const noexcept;
    [[nodiscard]] std::uint8_t highest_tile_exponent() const noexcept;
    [[nodiscard]] ExperimentMetrics metrics() const;
};

class GameRunner {
public:
    [[nodiscard]] static ExperimentResult run(Agent& agent, RunConfig config);

    // Plays exactly the same games as the single-agent overload, distributed
    // over `agents.size()` threads -- one agent per thread, because a
    // SearchAgent owns a transposition table and other mutable per-search
    // state that threads must not share.
    //
    // SCORES ARE UNCHANGED BY THE WORKER COUNT. Game i is always seeded
    // `first_seed + i` and is always played end to end by a single agent, and
    // no agent carries state across games that can affect play: SearchAgent's
    // transposition entries are stamped with a per-search generation that is
    // never reused, so an entry left by an earlier game can never be hit.
    // Games are handed out dynamically rather than in fixed blocks because
    // game lengths vary by more than 10x, and a static split would leave
    // workers idle. That is safe precisely because assignment cannot affect
    // the result. `GATE: parallel runner reproduces serial scores` in
    // tests/run_experiment_tests.cpp pins this.
    //
    // TIMING IS NOT. Per-game wall clock is measured under whatever contention
    // the other workers create, so `mean_milliseconds_per_move` from a
    // multi-worker run understates the agent's speed. Benchmark speed serially;
    // use workers to buy sample size. `worker_count` is recorded in the result
    // so a fast-but-parallel run can never be mistaken for a timing result.
    //
    // The evaluator behind every agent must be safe for concurrent reads.
    // Every evaluator in this project is (none holds mutable state), which is
    // checked by inspection rather than assumed -- see run_experiment_main.
    [[nodiscard]] static ExperimentResult run(
        const std::vector<Agent*>& agents, RunConfig config);
};

}  // namespace adversarial_2048
