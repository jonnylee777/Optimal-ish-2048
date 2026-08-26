#include "evaluation/h2_heuristic.hpp"

#include "core/board.hpp"

#include <array>

namespace adversarial_2048 {
namespace {

// The two orthogonal neighbors of a corner cell (row/column-adjacent, not
// diagonal), returned in unspecified order.
[[nodiscard]] constexpr std::array<std::size_t, 2> corner_neighbor_indices(
    std::size_t corner_index) noexcept {
    const auto row = corner_index / kBoardWidth;
    const auto column = corner_index % kBoardWidth;
    const auto row_neighbor =
        row == 0 ? corner_index + kBoardWidth : corner_index - kBoardWidth;
    const auto column_neighbor =
        column == 0 ? corner_index + 1 : corner_index - 1;
    return {row_neighbor, column_neighbor};
}

}  // namespace

double corner_chain_score(Board board) noexcept {
    const auto maximum = max_exponent(board);
    if (maximum == 0) {
        return 0.0;
    }
    const auto cells = decode(board);
    constexpr std::array<std::size_t, 4> corners{
        0, kBoardWidth - 1, kCellCount - kBoardWidth, kCellCount - 1};

    for (const auto corner : corners) {
        if (cells[corner] != maximum) {
            continue;
        }
        double score = 0.0;
        for (const auto neighbor : corner_neighbor_indices(corner)) {
            if (cells[neighbor] != 0 && cells[neighbor] == maximum - 1U) {
                score += static_cast<double>(cells[neighbor]);
            }
        }
        // Mirrors BaselineFeatures::corner_preference's convention of
        // crediting only the first matching corner, not summing over ties.
        return score;
    }
    return 0.0;
}

H2Features extract_h2_features(Board board) noexcept {
    return H2Features{extract_baseline_features(board), corner_chain_score(board)};
}

H2Heuristic::H2Heuristic(H2Weights weights) : weights_(weights) {}

double H2Heuristic::evaluate(Board board) const {
    const auto features = extract_h2_features(board);
    return weights_.baseline.empty_cells * features.baseline.empty_cells +
           weights_.baseline.monotonicity * features.baseline.monotonicity +
           weights_.baseline.smoothness * features.baseline.smoothness +
           weights_.baseline.corner_preference * features.baseline.corner_preference +
           weights_.corner_chain * features.corner_chain;
}

const H2Weights& H2Heuristic::weights() const noexcept {
    return weights_;
}

}  // namespace adversarial_2048
