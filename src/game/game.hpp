#pragma once

#include "core/board.hpp"
#include "core/random.hpp"
#include "core/spawn.hpp"

#include <cstdint>
#include <optional>

namespace adversarial_2048 {

struct TurnResult {
    bool moved{};
    std::uint64_t score_gained{};
    std::optional<SpawnEvent> spawn{};
};

class Game {
public:
    // Creates a standard game with two initial random tiles.
    explicit Game(std::uint64_t seed);

    // Creates a controlled state without initial spawns. This is useful for
    // deterministic tests and later competitive environments.
    Game(Board initial_board, std::uint64_t seed) noexcept;

    [[nodiscard]] Board board() const noexcept;
    [[nodiscard]] std::uint64_t score() const noexcept;
    [[nodiscard]] std::uint64_t move_count() const noexcept;
    [[nodiscard]] std::uint64_t seed() const noexcept;
    [[nodiscard]] bool game_over() const;

    // Invalid moves leave every part of the game state, including RNG state,
    // unchanged and do not spawn a tile.
    [[nodiscard]] TurnResult apply_move(Direction direction);

private:
    Board board_{};
    std::uint64_t score_{};
    std::uint64_t move_count_{};
    std::uint64_t seed_{};
    RandomEngine rng_{};
};

}  // namespace adversarial_2048
