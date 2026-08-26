#include "game/game.hpp"

#include <stdexcept>

namespace adversarial_2048 {

Game::Game(std::uint64_t seed)
    : seed_(seed), rng_(seed) {
    if (!spawn_random(board_, rng_).has_value() ||
        !spawn_random(board_, rng_).has_value()) {
        throw std::logic_error("failed to create the initial 2048 board");
    }
}

Game::Game(Board initial_board, std::uint64_t seed) noexcept
    : board_(initial_board), seed_(seed), rng_(seed) {}

Board Game::board() const noexcept {
    return board_;
}

std::uint64_t Game::score() const noexcept {
    return score_;
}

std::uint64_t Game::move_count() const noexcept {
    return move_count_;
}

std::uint64_t Game::seed() const noexcept {
    return seed_;
}

bool Game::game_over() const {
    return is_game_over(board_);
}

TurnResult Game::apply_move(Direction direction) {
    const auto result = move(board_, direction);
    if (!result.moved) {
        return {};
    }

    board_ = result.board;
    score_ += result.score;
    ++move_count_;
    const auto spawn = spawn_random(board_, rng_);
    if (!spawn.has_value()) {
        throw std::logic_error("a legal move did not leave room for a spawned tile");
    }
    return TurnResult{true, result.score, spawn};
}

}  // namespace adversarial_2048
