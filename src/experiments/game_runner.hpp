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

    [[nodiscard]] double mean_score() const noexcept;
    [[nodiscard]] std::uint64_t best_score() const noexcept;
    [[nodiscard]] double mean_moves() const noexcept;
    [[nodiscard]] std::uint8_t highest_tile_exponent() const noexcept;
    [[nodiscard]] ExperimentMetrics metrics() const;
};

class GameRunner {
public:
    [[nodiscard]] static ExperimentResult run(Agent& agent, RunConfig config);
};

}  // namespace adversarial_2048
