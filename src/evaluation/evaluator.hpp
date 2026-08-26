#pragma once

#include "core/board.hpp"

#include <cstdint>

namespace adversarial_2048 {

// Which kind of board an evaluator is meant to score. A 2048 transition is
//
//     state --(player move)--> AFTERSTATE --(random spawn)--> next state
//
// and the two are genuinely different distributions, so a value function
// trained on one is not meaningful on the other.
//
// The search consults this to decide what a leaf is. Both settings consume
// exactly one player decision layer per depth unit, so the project's depth
// semantics are unchanged.
enum class EvaluationSemantics : std::uint8_t {
    // Leaf is a board that already has the random tile on it. Depth 1 means
    // "move -> spawn -> evaluate". All hand-crafted heuristics (H0-H5).
    post_spawn_state,
    // Leaf is the board right after the player's move, before any spawn.
    // Depth 1 means "move -> evaluate afterstate". Learned afterstate value
    // functions (N-series) require this.
    afterstate,
};

class Evaluator {
public:
    virtual ~Evaluator() = default;
    [[nodiscard]] virtual double evaluate(Board board) const = 0;

    // Defaults to post-spawn so every existing evaluator keeps its behaviour.
    [[nodiscard]] virtual EvaluationSemantics semantics() const noexcept {
        return EvaluationSemantics::post_spawn_state;
    }
    [[nodiscard]] virtual double evaluate_transition(Board, Board) const {
        return 0.0;
    }

    // What a TERMINAL position (no legal move) is worth to this evaluator.
    //
    // The search must not ask evaluate() about a dead board. A value function
    // that predicts remaining score answers "0" -- the game is over. A
    // positional heuristic has no such number, and measurement shows H0/H4/H5
    // all return negative values on bad boards (down to -6.3e6 for H4), so
    // substituting 0 would rank DEATH above many bad-but-alive positions.
    //
    // Default therefore is "worse than anything achievable". Finite rather than
    // -infinity so it survives being added to accumulated rewards without
    // producing NaN.
    //
    // Getting this wrong cost this project a factor of 1.7 in playing strength
    // at depth 3 -- see E21 in docs/ULTIMATE_AGENT_PROGRESS.md.
    [[nodiscard]] virtual double terminal_value() const noexcept {
        return -1e15;
    }

    // True only if evaluate()/evaluate_transition() are invariant under all
    // 8 board rotations/reflections. Search may use this to safely enable
    // dihedral-symmetry transposition-table canonicalization. Defaults to
    // false (the safe assumption) since most structural/anchored heuristics
    // are not invariant.
    [[nodiscard]] virtual bool is_rotation_invariant() const noexcept {
        return false;
    }
};

}  // namespace adversarial_2048
