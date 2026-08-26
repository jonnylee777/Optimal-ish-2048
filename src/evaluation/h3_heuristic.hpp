#pragma once

#include "evaluation/evaluator.hpp"
#include "evaluation/h2_heuristic.hpp"
#include "evaluation/structural_heuristic.hpp"

namespace adversarial_2048 {

// H3 = H2 (all five H2 features) + the snake/main-line structural features
// already implemented and tested as the legacy StructuralHeuristic
// (src/evaluation/structural_heuristic.hpp): a continuous reward for
// ordering tiles in a decreasing snake path anchored at one board corner,
// a stability bonus, an adverse-stuck-state penalty, and a move-transition
// penalty for displacing the anchored high-value prefix. Reuses
// StructuralHeuristic's public feature functions rather than duplicating
// them — legacy code is untouched. See docs/experiment-taxonomy.md.
struct H3Features {
    H2Features h2;
    double main_line{};
    double structural_stability{};
    double adverse_stuck_penalty{};
};

struct H3Weights {
    H2Weights h2{};
    // Same initial hand-picked, unoptimized values as legacy StructuralWeights.
    double main_line{4.0};
    double structural_stability{1.0};
    double adverse_stuck{2.0};
    double structural_displacement{2.0};
};

[[nodiscard]] H3Features extract_h3_features(
    Board board, BoardCorner corner = BoardCorner::bottom_right) noexcept;

class H3Heuristic final : public Evaluator {
public:
    explicit H3Heuristic(
        H3Weights weights = {}, BoardCorner corner = BoardCorner::bottom_right);

    [[nodiscard]] double evaluate(Board board) const override;
    [[nodiscard]] double evaluate_transition(Board old_board, Board new_board) const override;
    // Corner-anchored, same as StructuralHeuristic: not safe for dihedral
    // symmetry canonicalization.
    [[nodiscard]] bool is_rotation_invariant() const noexcept override {
        return false;
    }
    [[nodiscard]] const H3Weights& weights() const noexcept;
    [[nodiscard]] BoardCorner corner() const noexcept;

private:
    H3Weights weights_;
    BoardCorner corner_;
};

}  // namespace adversarial_2048
