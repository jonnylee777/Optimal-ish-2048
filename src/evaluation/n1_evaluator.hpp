#pragma once

#include "evaluation/evaluator.hpp"
#include "learning/ntuple_network.hpp"

namespace adversarial_2048 {

// N1: the first *learned* evaluator in this project. Everything in the
// H-series (H0-H5) is hand-crafted — either our own features or a
// transcription of someone else's tuned constants. N1's weights come from
// self-play temporal-difference learning with no human 2048 knowledge
// encoded at all.
//
// This is a thin adapter: the network does the work (see
// learning/ntuple_network.hpp), and this just presents it through the
// Evaluator interface so it can be dropped into the existing Expectimax
// search alongside any other heuristic.
//
// Holds the network by reference and does not own it, matching how every
// other evaluator here is constructed on the stack at the call site and
// handed to the search as a `const Evaluator&`. The referenced network must
// outlive this object.
class N1Evaluator final : public Evaluator {
public:
    explicit N1Evaluator(const learning::NTupleNetwork& network) : network_(network) {}

    [[nodiscard]] double evaluate(Board board) const override {
        return network_.value(board);
    }

    // Symmetric weight sharing (each tuple's 8 dihedral orderings index the
    // same table) makes the value invariant under all 8 board symmetries by
    // construction. Verified empirically in tests/ntuple_network_tests.cpp
    // rather than merely asserted.
    [[nodiscard]] bool is_rotation_invariant() const noexcept override { return true; }

    // N1 is an AFTERSTATE value function: trained by
    // V(afterstate) <- reward_of_next_move + V(next afterstate), so it answers
    // "expected future score from this post-move, pre-spawn board". Evaluating
    // it on post-spawn states is not a well-defined quantity and collapses
    // playing strength — see docs/ntuple-learning.md.
    [[nodiscard]] EvaluationSemantics semantics() const noexcept override {
        return EvaluationSemantics::afterstate;
    }

    // This network predicts REMAINING SCORE, so a terminal position is worth
    // exactly zero more points. Not a sentinel -- the true value.
    [[nodiscard]] double terminal_value() const noexcept override { return 0.0; }

    [[nodiscard]] const learning::NTupleNetwork& network() const noexcept { return network_; }

private:
    const learning::NTupleNetwork& network_;
};

}  // namespace adversarial_2048
