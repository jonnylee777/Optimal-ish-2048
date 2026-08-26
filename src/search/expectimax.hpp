#pragma once

#include "core/board.hpp"
#include "evaluation/evaluator.hpp"

#include <cstddef>
#include <chrono>
#include <cstdint>
#include <optional>
#include <vector>

namespace adversarial_2048 {

struct SearchStatistics {
    std::uint64_t player_nodes{};
    std::uint64_t chance_nodes{};
    std::uint64_t leaf_evaluations{};
    std::uint64_t spawn_outcomes{};
    std::uint64_t cache_lookups{};
    std::uint64_t cache_hits{};
    double elapsed_seconds{};

    [[nodiscard]] std::uint64_t total_nodes() const noexcept;
    [[nodiscard]] double cache_hit_rate() const noexcept;
    [[nodiscard]] double nodes_per_second() const noexcept;
};

struct SearchResult {
    std::optional<Direction> direction;
    double value{};
    SearchStatistics statistics;
    std::uint32_t completed_depth{};
};

struct ExpectimaxOptions {
    bool use_transposition_table{true};
    std::size_t transposition_table_capacity{1U << 16U};
    // Zero keeps exact Expectimax. A positive value evaluates chance outcomes
    // whose cumulative path probability is smaller than this threshold.
    double minimum_path_probability{0.0};
    // Safe only when the evaluator and transition value are invariant under
    // rotations and reflections. The four-part baseline satisfies this.
    bool use_symmetry_reduction{false};
    // Zero requires the requested depth to complete. A positive budget uses
    // iterative deepening and returns the last fully completed depth.
    double time_limit_seconds{0.0};
};

class Expectimax {
public:
    explicit Expectimax(const Evaluator& evaluator, ExpectimaxOptions options = {});

    // Depth is the number of player decision layers. Depth 1 performs:
    // player move -> all random spawns -> evaluator.
    [[nodiscard]] SearchResult search(Board board, std::uint32_t depth);

    // Value of the subtree rooted at an AFTERSTATE (post-move, pre-spawn),
    // which is what a root move leads to. Exposed so several root moves can be
    // explored concurrently, each by its own Expectimax with its own
    // transposition table — the tables are per-instance mutable state, so
    // sharing one across threads would be a data race.
    [[nodiscard]] double afterstate_value(Board afterstate, std::uint32_t depth);
    [[nodiscard]] SearchResult search_iterative(Board board, std::uint32_t maximum_depth);

private:
    enum class NodeType : std::uint8_t {
        player,
        chance,
    };

    struct CacheKey {
        Board board;
        std::uint32_t depth{};
        NodeType node_type{};
        std::uint64_t probability_bits{};

        [[nodiscard]] bool operator==(const CacheKey&) const = default;
    };

    struct CacheKeyHash {
        [[nodiscard]] std::size_t operator()(const CacheKey& key) const noexcept;
    };

    struct CacheEntry {
        CacheKey key{};
        double value{};
        std::uint32_t generation{};
    };

    [[nodiscard]] double player_value(
        Board board, std::uint32_t depth, double path_probability);
    [[nodiscard]] double chance_value(
        Board board, std::uint32_t depth, double path_probability);
    [[nodiscard]] double leaf_value(Board board);
    [[nodiscard]] SearchResult search_current_generation(Board board, std::uint32_t depth);
    void begin_search(bool bounded);
    void check_deadline();
    [[nodiscard]] CacheKey cache_key(
        Board board, std::uint32_t depth, NodeType node_type,
        double path_probability) const noexcept;
    [[nodiscard]] std::optional<double> cached_value(const CacheKey& key);
    void store_value(const CacheKey& key, double value);

    const Evaluator& evaluator_;
    ExpectimaxOptions options_;
    std::vector<CacheEntry> cache_;
    std::uint32_t cache_generation_{};
    SearchStatistics statistics_{};
    std::chrono::steady_clock::time_point search_start_{};
    std::chrono::steady_clock::time_point deadline_{};
    std::uint64_t deadline_node_counter_{};
};

}  // namespace adversarial_2048
