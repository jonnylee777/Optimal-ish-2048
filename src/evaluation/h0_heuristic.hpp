#pragma once

#include "evaluation/evaluator.hpp"

namespace adversarial_2048 {

// H0: the simplest heuristic in this codebase, modeled on the original
// nneonneo heuristic described in reference.md — "bonuses for open squares
// and for having large values on the edge" — before monotonicity/smoothness
// were introduced. Exists as a genuine floor below H1 (BaselineHeuristic),
// not an arbitrary toy. See docs/phase1-heuristics.md.
struct H0Features {
    double empty_cells{};
    double edge_max_bonus{};
};

// Deliberately hand-picked, unoptimized weights, same precedent as H1's
// initial human-selected weights.
struct H0Weights {
    double empty_cells{100.0};
    double edge_max_bonus{50.0};
};

[[nodiscard]] H0Features extract_h0_features(Board board) noexcept;

class H0Heuristic final : public Evaluator {
public:
    explicit H0Heuristic(H0Weights weights = {});

    [[nodiscard]] double evaluate(Board board) const override;
    [[nodiscard]] bool is_rotation_invariant() const noexcept override {
        return true;
    }
    [[nodiscard]] const H0Weights& weights() const noexcept;

private:
    H0Weights weights_;
};

}  // namespace adversarial_2048
