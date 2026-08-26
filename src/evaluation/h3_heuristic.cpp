#include "evaluation/h3_heuristic.hpp"

namespace adversarial_2048 {

H3Features extract_h3_features(Board board, BoardCorner corner) noexcept {
    const auto structural = extract_structural_features(board, corner);
    H3Features features;
    features.h2.baseline = structural.baseline;
    features.h2.corner_chain = corner_chain_score(board);
    features.main_line = structural.main_line;
    features.structural_stability = structural.structural_stability;
    features.adverse_stuck_penalty = structural.adverse_stuck_penalty;
    return features;
}

H3Heuristic::H3Heuristic(H3Weights weights, BoardCorner corner)
    : weights_(weights), corner_(corner) {}

double H3Heuristic::evaluate(Board board) const {
    const auto features = extract_h3_features(board, corner_);
    return weights_.h2.baseline.empty_cells * features.h2.baseline.empty_cells +
           weights_.h2.baseline.monotonicity * features.h2.baseline.monotonicity +
           weights_.h2.baseline.smoothness * features.h2.baseline.smoothness +
           weights_.h2.baseline.corner_preference * features.h2.baseline.corner_preference +
           weights_.h2.corner_chain * features.h2.corner_chain +
           weights_.main_line * features.main_line +
           weights_.structural_stability * features.structural_stability -
           weights_.adverse_stuck * features.adverse_stuck_penalty;
}

double H3Heuristic::evaluate_transition(Board old_board, Board new_board) const {
    if (weights_.structural_displacement == 0.0) {
        return 0.0;
    }
    const auto orientation = extract_structural_features(old_board, corner_).selected_orientation;
    return -weights_.structural_displacement *
           structural_movement_penalty(old_board, new_board, corner_, orientation);
}

const H3Weights& H3Heuristic::weights() const noexcept {
    return weights_;
}

BoardCorner H3Heuristic::corner() const noexcept {
    return corner_;
}

}  // namespace adversarial_2048
