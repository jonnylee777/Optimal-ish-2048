#pragma once

#include "evaluation/baseline_heuristic.hpp"
#include "evaluation/evaluator.hpp"

namespace adversarial_2048 {

// H2 = H1 (all four BaselineFeatures, reused as-is) + "corner chain": H1's
// corner_preference is all-or-nothing (full credit only if the max tile sits
// exactly in a corner, nothing about what surrounds it). Corner chain
// additionally rewards the max tile's two orthogonal neighbors for holding
// exactly the next-largest value, reinforcing the classic "build a
// descending chain out from the corner" strategy. Isolates one change from
// H1 for a clean comparison. See docs/phase1-heuristics.md.
struct H2Features {
    BaselineFeatures baseline;
    double corner_chain{};
};

struct H2Weights {
    // Reuses H1's own initial human-selected weights as the default, so
    // toggling corner_chain to 0 exactly reproduces H1's behavior.
    BaselineWeights baseline{};
    // Hand-picked, unoptimized, same precedent as H1's and H0's initial weights.
    double corner_chain{60.0};
};

// The "corner chain" feature in isolation — exposed so other heuristics
// (e.g. H3) can compose it without duplicating the computation.
[[nodiscard]] double corner_chain_score(Board board) noexcept;

[[nodiscard]] H2Features extract_h2_features(Board board) noexcept;

class H2Heuristic final : public Evaluator {
public:
    explicit H2Heuristic(H2Weights weights = {});

    [[nodiscard]] double evaluate(Board board) const override;
    [[nodiscard]] bool is_rotation_invariant() const noexcept override {
        return true;
    }
    [[nodiscard]] const H2Weights& weights() const noexcept;

private:
    H2Weights weights_;
};

}  // namespace adversarial_2048
