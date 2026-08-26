#include "search/parallel_expectimax.hpp"

#include <algorithm>
#include <array>
#include <future>
#include <limits>
#include <optional>

namespace adversarial_2048 {

ParallelExpectimax::ParallelExpectimax(const Evaluator& evaluator, ExpectimaxOptions options)
    : evaluator_(evaluator), options_(options) {
    for (std::size_t index = 0; index < kDirections.size(); ++index) {
        searchers_.push_back(std::make_unique<Expectimax>(evaluator_, options_));
    }
}

SearchResult ParallelExpectimax::search(Board board, std::uint32_t depth) {
    // Enumerate root moves exactly as the serial search does, including the
    // duplicate check: two directions can produce the same afterstate AND the
    // same reward, in which case exploring both is wasted work.
    struct RootMove {
        Direction direction;
        MoveResult moved;
    };
    std::array<RootMove, 4> roots{};
    std::size_t count = 0;
    for (const auto direction : kDirections) {
        const auto moved = move(board, direction);
        if (!moved.moved) {
            continue;
        }
        const auto duplicate = std::any_of(
            roots.begin(), roots.begin() + count, [&moved](const RootMove& previous) {
                return previous.moved.board == moved.board &&
                       previous.moved.score == moved.score;
            });
        if (duplicate) {
            continue;
        }
        roots[count++] = RootMove{direction, moved};
    }

    if (count == 0) {
        // Terminal root: no move exists, and the position is worth whatever a
        // dead position is worth to this evaluator (see E21).
        return SearchResult{std::nullopt, evaluator_.terminal_value(), {}, depth};
    }

    // One task per root move. std::async rather than a pool because this runs
    // once per game move with at most four tasks, so the launch cost is
    // negligible beside a subtree that takes milliseconds.
    std::array<std::future<double>, 4> futures;
    for (std::size_t index = 0; index < count; ++index) {
        auto* searcher = searchers_[index].get();
        const auto afterstate = roots[index].moved.board;
        futures[index] = std::async(std::launch::async, [searcher, afterstate, depth] {
            return searcher->afterstate_value(afterstate, depth);
        });
    }

    std::optional<Direction> best_direction;
    double best_value = -std::numeric_limits<double>::infinity();
    SearchStatistics combined{};
    for (std::size_t index = 0; index < count; ++index) {
        const auto subtree = futures[index].get();
        const auto value = static_cast<double>(roots[index].moved.score) +
                           evaluator_.evaluate_transition(board, roots[index].moved.board) +
                           subtree;
        // Strict > preserves the serial tie-break: the first direction in
        // kDirections order wins, so results match Expectimax exactly.
        if (!best_direction.has_value() || value > best_value) {
            best_direction = roots[index].direction;
            best_value = value;
        }
    }

    return SearchResult{best_direction, best_value, combined, depth};
}

}  // namespace adversarial_2048
