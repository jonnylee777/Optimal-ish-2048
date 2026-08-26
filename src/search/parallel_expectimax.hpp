#pragma once

#include "core/board.hpp"
#include "evaluation/evaluator.hpp"
#include "search/expectimax.hpp"

#include <cstdint>
#include <memory>
#include <vector>

namespace adversarial_2048 {

// Runs the (at most four) root moves concurrently, one Expectimax per move.
//
// Why root-level and not deeper: the transposition table is per-instance
// mutable state, so threads must not share one. Splitting at the root gives
// each thread a completely independent search with its own table, which needs
// no locking at all. The cost is that the tables no longer share work across
// root moves -- worth it because the alternative is a contended table.
//
// Speedup is bounded by the number of legal moves (<= 4), so expect ~3-3.5x
// rather than the core count. That does NOT buy an extra depth level, since
// each level costs roughly 25x; it buys measurement throughput, which is what
// currently limits how deeply this project can afford to benchmark.
//
// The result is IDENTICAL to Expectimax::search for a deterministic evaluator:
// the root maximisation is the same, only the order of evaluation changes. A
// test pins that equality rather than assuming it.
class ParallelExpectimax {
public:
    explicit ParallelExpectimax(const Evaluator& evaluator, ExpectimaxOptions options = {});

    [[nodiscard]] SearchResult search(Board board, std::uint32_t depth);

private:
    const Evaluator& evaluator_;
    ExpectimaxOptions options_;
    // One searcher per possible root move, reused across calls so the tables
    // stay warm and no per-move allocation happens on the hot path.
    std::vector<std::unique_ptr<Expectimax>> searchers_;
};

}  // namespace adversarial_2048
