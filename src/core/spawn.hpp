#pragma once

#include "core/board.hpp"
#include "core/random.hpp"

#include <cstddef>
#include <cstdint>
#include <optional>

namespace adversarial_2048 {

struct SpawnEvent {
    std::size_t cell_index{};
    std::uint8_t exponent{};
};

// Selects an empty cell uniformly, then spawns a 2 with probability 0.9 or a
// 4 with probability 0.1. The caller owns and seeds the random engine.
[[nodiscard]] std::optional<SpawnEvent> spawn_random(Board& board, RandomEngine& rng);

}  // namespace adversarial_2048
