#pragma once

#include "evaluation/evaluator.hpp"
#include "learning/ntuple_network.hpp"

#include <cstddef>
#include <memory>
#include <utility>
#include <vector>

namespace adversarial_2048 {

// Averages the value estimates of several independently-trained networks.
//
// Motivation, and why this is not another variant of what came before. Every
// previous improvement here — global features, backward replay, optimistic
// initialisation, deeper search — corrected the same defect (the value function
// misjudging positions near the end of a game), which is why they turned out to
// be substitutes for one another rather than additive (E13, E25).
//
// An ensemble attacks something different: the *variance* of V. Playing at
// 1 ply consumes only the ORDERING of V, and ordering quality is limited by how
// noisy each estimate is — the correlation between V and true remaining score
// is about 0.56 for the best network here. Averaging k independent estimators
// with uncorrelated errors cuts the error standard deviation by sqrt(k) without
// touching any particular defect.
//
// The condition for that to work is that the members must fail on DIFFERENT
// positions. Measured before building this: the per-seed score correlation
// between the plain and global-feature networks at depth 4 is **0.042** —
// essentially independent. That is what makes averaging worth trying rather
// than merely plausible.
//
// Cost is linear in members: k networks means k times the evaluation work and
// k times the resident memory. At depth 4 a 2-member ensemble therefore costs
// about what depth 4.3 would, so it has to earn its keep against simply
// searching deeper.
class EnsembleEvaluator final : public Evaluator {
public:
    // Takes ownership so callers do not have to keep k networks alive
    // separately. Throws std::invalid_argument on an empty ensemble.
    explicit EnsembleEvaluator(
        std::vector<std::unique_ptr<learning::NTupleNetwork>> networks);

    [[nodiscard]] double evaluate(Board board) const override;

    // Every member is an afterstate value function, so the ensemble is too.
    [[nodiscard]] EvaluationSemantics semantics() const noexcept override {
        return EvaluationSemantics::afterstate;
    }

    // A mean of predicted-remaining-score functions is still a predicted
    // remaining score, so a terminal position is worth exactly zero.
    [[nodiscard]] double terminal_value() const noexcept override { return 0.0; }

    // Each member is invariant under the 8 board symmetries by construction
    // (dihedral weight sharing), so their mean is too. Note this is declared
    // but NOT worth acting on: symmetry reduction measured 60% slower than it
    // saves for this evaluator family (E24).
    [[nodiscard]] bool is_rotation_invariant() const noexcept override { return true; }

    [[nodiscard]] std::size_t member_count() const noexcept { return networks_.size(); }

private:
    std::vector<std::unique_ptr<learning::NTupleNetwork>> networks_;
    double inverse_count_{};
};

}  // namespace adversarial_2048
