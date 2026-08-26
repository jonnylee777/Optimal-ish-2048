#include "agents/greedy_agent.hpp"

#include <cstdint>

namespace adversarial_2048 {

std::string_view GreedyAgent::name() const noexcept {
    return "greedy";
}

void GreedyAgent::reset(std::uint64_t) {}

std::optional<Direction> GreedyAgent::choose_move(Board board) {
    std::optional<Direction> best_direction;
    std::uint64_t best_score = 0;

    for (const auto direction : kDirections) {
        const auto result = move(board, direction);
        if (result.moved && (!best_direction.has_value() || result.score > best_score)) {
            best_direction = direction;
            best_score = result.score;
        }
    }
    return best_direction;
}

}  // namespace adversarial_2048
