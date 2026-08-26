#pragma once

#include "evaluation/evaluator.hpp"

namespace adversarial_2048 {

// H4: a faithful port of the reference nneonneo/2048-ai heuristic
// (https://github.com/nneonneo/2048-ai, MIT licensed, Copyright 2014-2019
// Robert Xiao and contributors), described in ../reference.md. That
// implementation reports reaching 32768 in ~36% of games.
//
// NOTE: unlike H1->H2->H3, H4 does NOT extend H3 — it is a standalone
// heuristic with its own feature set, ported wholesale rather than composed
// on top of ours. It is numbered H4 simply as the next slot. Its four
// features are computed per 16-bit row and applied to all four rows plus
// all four columns (via transpose), so each feature is a sum of 8 row
// lookups.
//
// Feature definitions (fixed; the exponents below are part of the feature,
// not tunable weights):
//   empty        - count of empty cells in the row
//   merges       - adjacent-equal runs, counted as (1 + run length)
//   monotonicity - min(left-descending, right-descending) cost, where each
//                  step costs pow(rank, 4). Entered as a POSITIVE cost and
//                  subtracted at evaluation time.
//   sum          - sum of pow(rank, 3.5) over the row. Also a positive cost,
//                  subtracted at evaluation time.
struct H4Features {
    double empty{};
    double merges{};
    double monotonicity{};
    double sum{};
};

// The reference implementation's own CMA-ES-optimized weights. Kept exactly
// as published so H4's default is a reproduction, not a re-guess.
struct H4Weights {
    double empty{270.0};
    double merges{700.0};
    double monotonicity{47.0};
    double sum{11.0};
};

// Reference's per-row SCORE_LOST_PENALTY, summed over the 8 row/column
// lookups. A constant offset that keeps scores positive; it cannot change
// any move ranking, and is included only so H4's absolute values match the
// reference implementation's.
inline constexpr double kH4LostPenaltyTotal = 200'000.0 * 8.0;

[[nodiscard]] H4Features extract_h4_features(Board board) noexcept;

class H4Heuristic final : public Evaluator {
public:
    explicit H4Heuristic(H4Weights weights = {});

    [[nodiscard]] double evaluate(Board board) const override;
    // Every feature is computed per row and summed over rows and columns
    // alike, and monotonicity takes min(left, right) — so the value is
    // unchanged by any rotation or reflection.
    [[nodiscard]] bool is_rotation_invariant() const noexcept override {
        return true;
    }
    [[nodiscard]] const H4Weights& weights() const noexcept;

private:
    H4Weights weights_;
};

}  // namespace adversarial_2048
