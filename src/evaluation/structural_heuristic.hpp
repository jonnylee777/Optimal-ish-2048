#pragma once

#include "evaluation/baseline_heuristic.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>

namespace adversarial_2048 {

enum class BoardCorner : std::uint8_t {
    top_left,
    top_right,
    bottom_left,
    bottom_right,
};

enum class SnakeOrientation : std::uint8_t {
    horizontal_first,
    vertical_first,
};

using SnakePath = std::array<std::size_t, kCellCount>;

struct MainLineFrontier {
    std::size_t path_index{};
    std::size_t cell_index{};
    std::size_t row{};
    std::size_t column{};
    std::uint8_t current_exponent{};
    std::uint8_t desired_exponent{};
};

struct StuckStateClassification {
    bool is_stuck{};
    bool is_adverse{};
    std::array<bool, 4> legal_moves{};
    std::array<bool, 4> safe_moves{};
    std::array<bool, 4> escape_moves{};
    std::array<bool, 4> merge_moves{};
    std::array<std::size_t, 4> exposed_prefix_positions{};
    std::size_t legal_move_count{};
    std::size_t safe_move_count{};
    std::size_t merge_move_count{};
    double minimum_displacement{};
};

struct StructuralFeatures {
    BaselineFeatures baseline;
    double main_line{};
    double structural_stability{};
    double adverse_stuck_penalty{};
    std::size_t anchored_prefix_length{};
    SnakeOrientation selected_orientation{SnakeOrientation::horizontal_first};
};

struct StructuralWeights {
    BaselineWeights baseline{kDepth3OptimizedBaselineWeights};
    // Initial experimental values, not optimized weights.
    double main_line{4.0};
    double structural_stability{1.0};
    double adverse_stuck{2.0};
    double structural_displacement{2.0};
};

// Candidate 6 from the exact depth-4 joint optimization on training seeds
// 1200-1219. It must be treated as a frozen candidate until validation.
inline constexpr StructuralWeights kDepth4OptimizedStructuralWeights{
    {115.82974924638357,
     37.30236719801037,
     24.712293838159923,
     27.468659832176016},
    1.3726429758766383,
    1.17441510554765,
    3.0596792676181983,
    0.5608988108966877,
};

[[nodiscard]] SnakePath get_snake_path(
    BoardCorner corner, SnakeOrientation orientation) noexcept;
[[nodiscard]] double main_line_score(
    Board board, BoardCorner corner, SnakeOrientation orientation) noexcept;
[[nodiscard]] std::size_t anchored_main_line_length(
    Board board, BoardCorner corner, SnakeOrientation orientation) noexcept;
[[nodiscard]] std::optional<MainLineFrontier> main_line_frontier(
    Board board, BoardCorner corner, SnakeOrientation orientation) noexcept;
[[nodiscard]] double structural_movement_penalty(
    Board old_board, Board new_board, BoardCorner corner,
    SnakeOrientation orientation) noexcept;
[[nodiscard]] double structural_stability_score(
    Board board, BoardCorner corner, SnakeOrientation orientation);
[[nodiscard]] StuckStateClassification classify_stuck_state(
    Board board, BoardCorner corner, SnakeOrientation orientation);
[[nodiscard]] StructuralFeatures extract_structural_features(
    Board board, BoardCorner corner = BoardCorner::bottom_right);

class StructuralHeuristic final : public Evaluator {
public:
    explicit StructuralHeuristic(
        StructuralWeights weights = {},
        BoardCorner corner = BoardCorner::bottom_right);

    [[nodiscard]] double evaluate(Board board) const override;
    [[nodiscard]] double evaluate_transition(Board old_board, Board new_board) const override;
    [[nodiscard]] const StructuralWeights& weights() const noexcept;
    [[nodiscard]] BoardCorner corner() const noexcept;

private:
    StructuralWeights weights_;
    BoardCorner corner_;
};

[[nodiscard]] StructuralWeights main_line_ablation_weights() noexcept;
[[nodiscard]] StructuralWeights movement_ablation_weights() noexcept;

}  // namespace adversarial_2048
