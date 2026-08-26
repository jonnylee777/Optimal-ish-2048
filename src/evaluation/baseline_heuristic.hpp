#pragma once

#include "evaluation/evaluator.hpp"

namespace adversarial_2048 {

struct BaselineFeatures {
    double empty_cells{};
    double monotonicity{};
    double smoothness{};
    double corner_preference{};
};

// These are deliberately initial, human-selected weights—not optimized
// values. Feature definitions remain fixed when weights are optimized later.
struct BaselineWeights {
    double empty_cells{270.0};
    double monotonicity{47.0};
    double smoothness{15.0};
    double corner_preference{100.0};
};

// Selected by deterministic multi-stage random search at depth 3, then chosen
// from finalists on validation seeds 10000-10024. The original defaults above
// remain unchanged so both policy generations can always be compared.
inline constexpr BaselineWeights kDepth3OptimizedBaselineWeights{
    85.221621525239641,
    44.83062947862652,
    43.676658929747916,
    28.751796341054963,
};

[[nodiscard]] BaselineFeatures extract_baseline_features(Board board) noexcept;

class BaselineHeuristic final : public Evaluator {
public:
    explicit BaselineHeuristic(BaselineWeights weights = {});

    [[nodiscard]] double evaluate(Board board) const override;
    [[nodiscard]] bool is_rotation_invariant() const noexcept override {
        return true;
    }
    [[nodiscard]] const BaselineWeights& weights() const noexcept;

private:
    BaselineWeights weights_;
};

}  // namespace adversarial_2048
