#include "agents/random_agent.hpp"

#include <array>
#include <cstddef>

namespace adversarial_2048 {

RandomAgent::RandomAgent(std::uint64_t seed)
    : rng_(seed) {}

std::string_view RandomAgent::name() const noexcept {
    return "random";
}

void RandomAgent::reset(std::uint64_t seed) {
    rng_.seed(seed);
}

std::optional<Direction> RandomAgent::choose_move(Board board) {
    std::array<Direction, kDirections.size()> available{};
    std::size_t count = 0;
    for (const auto direction : kDirections) {
        if (can_move(board, direction)) {
            available[count++] = direction;
        }
    }

    if (count == 0) {
        return std::nullopt;
    }
    return available[static_cast<std::size_t>(sample_bounded(rng_, count))];
}

}  // namespace adversarial_2048
