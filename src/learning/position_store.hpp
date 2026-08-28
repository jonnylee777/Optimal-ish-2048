#pragma once

#include "core/board.hpp"

#include <cstdint>
#include <filesystem>
#include <vector>

namespace adversarial_2048::learning {

// A flat file of board positions, used to start training episodes somewhere
// other than an empty board.
//
// Why this exists. Training is 1-ply greedy self-play; the deployed agent plays
// at depth 4. Measured, those are very different distributions: at depth 1 the
// agent reaches 16384 in 54% of games and 32768 in 0%, while at depth 4 it
// reaches 16384 in 97%. An autopsy of 40 depth-4 games found **38 of them died
// having never assembled a second 16384** — not jammed, not one merge short.
// So the skill that decides every game is "hold a 16384 in the corner and
// rebuild the snake beneath it", and the value function currently receives
// almost no updates in that regime because self-play rarely goes there.
//
// Seeding episodes from collected late-game positions changes WHICH states get
// updates rather than how many, which is the one thing more training does not
// do — 1M -> 2M games gained 4.7% at depth 1 and nothing at depth 4.
//
// Format is deliberately trivial (magic, count, then packed pairs): these files
// are regenerable in minutes and never need to outlive a experiment.
void save_positions(const std::filesystem::path& path, const std::vector<Board>& positions);

[[nodiscard]] std::vector<Board> load_positions(const std::filesystem::path& path);

// A position paired with the value a DEEP SEARCH assigned to it.
//
// Used for distillation: rather than bootstrapping V from its own 1-ply
// estimates, fit it directly to what depth-4 expectimax concluded. That is a
// different training signal, not another feature — and doing the search offline
// avoids the 34x cost that made searched training useless when tried inline.
struct ValuedPosition {
    Board board;
    float target{};
};

[[nodiscard]] std::vector<ValuedPosition> load_valued_positions(
    const std::filesystem::path& path);

}  // namespace adversarial_2048::learning
