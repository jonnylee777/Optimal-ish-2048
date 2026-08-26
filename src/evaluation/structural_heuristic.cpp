#include "evaluation/structural_heuristic.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

namespace adversarial_2048 {
namespace {

constexpr std::uint8_t kMinimumProtectedExponent = 4;

[[nodiscard]] constexpr bool is_bottom(BoardCorner corner) noexcept {
    return corner == BoardCorner::bottom_left || corner == BoardCorner::bottom_right;
}

[[nodiscard]] constexpr bool is_right(BoardCorner corner) noexcept {
    return corner == BoardCorner::top_right || corner == BoardCorner::bottom_right;
}

[[nodiscard]] constexpr double position_weight(std::size_t index) noexcept {
    constexpr std::array<double, kCellCount> weights{
        1.0000, 0.8500, 0.7225, 0.6141,
        0.5220, 0.4437, 0.3771, 0.3206,
        0.2725, 0.2316, 0.1969, 0.1673,
        0.1422, 0.1209, 0.1028, 0.0874,
    };
    return weights[index];
}

[[nodiscard]] double adjacency_score(
    int current, int next, std::size_t index) noexcept {
    if (current == 0) {
        return 0.0;
    }
    const auto importance = position_weight(index) * static_cast<double>(current);
    if (next == 0) {
        return -0.5 * importance;
    }
    const auto gap = current - next;
    if (gap == 1) {
        return 2.0 * importance;
    }
    auto score = -importance * static_cast<double>(std::abs(gap - 1));
    if (gap < 0) {
        score -= importance * static_cast<double>(-gap);
    }
    return score;
}

class MainLineTables {
public:
    static constexpr std::size_t kRowCount = 1U << 16U;

    MainLineTables() {
        for (std::size_t block = 0; block < 4; ++block) {
            for (std::size_t value = 0; value < kRowCount; ++value) {
                std::array<int, 4> exponents{};
                double score = 0.0;
                for (std::size_t offset = 0; offset < 4; ++offset) {
                    exponents[offset] = static_cast<int>(
                        (value >> (offset * 4U)) & 0xFU);
                    const auto exponent = static_cast<double>(exponents[offset]);
                    score += position_weight(block * 4 + offset) * exponent * exponent;
                }
                for (std::size_t offset = 0; offset < 3; ++offset) {
                    score += adjacency_score(
                        exponents[offset], exponents[offset + 1],
                        block * 4 + offset);
                }
                scores_[block][value] = score;
            }
        }
    }

    [[nodiscard]] double score(std::size_t block, std::uint16_t row) const noexcept {
        return scores_[block][row];
    }

private:
    std::array<std::array<double, kRowCount>, 4> scores_{};
};

[[nodiscard]] const MainLineTables& main_line_tables() {
    static const MainLineTables tables;
    return tables;
}

struct SelectedLine {
    SnakeOrientation orientation{SnakeOrientation::horizontal_first};
    double score{};
};

[[nodiscard]] SelectedLine select_main_line(Board board, BoardCorner corner) noexcept {
    const auto horizontal = main_line_score(
        board, corner, SnakeOrientation::horizontal_first);
    const auto vertical = main_line_score(
        board, corner, SnakeOrientation::vertical_first);
    return vertical > horizontal
        ? SelectedLine{SnakeOrientation::vertical_first, vertical}
        : SelectedLine{SnakeOrientation::horizontal_first, horizontal};
}

[[nodiscard]] std::size_t protected_prefix_length(
    Board board, BoardCorner corner, SnakeOrientation orientation) noexcept {
    const auto path = get_snake_path(corner, orientation);
    const auto cells = decode(board);
    const auto anchored = anchored_main_line_length(board, corner, orientation);
    std::size_t protected_length = 0;
    while (protected_length < anchored &&
           cells[path[protected_length]] >= kMinimumProtectedExponent) {
        ++protected_length;
    }
    return protected_length;
}

[[nodiscard]] bool preferred_direction(BoardCorner corner, Direction direction) noexcept {
    if (direction == Direction::left) {
        return !is_right(corner);
    }
    if (direction == Direction::right) {
        return is_right(corner);
    }
    if (direction == Direction::up) {
        return !is_bottom(corner);
    }
    return is_bottom(corner);
}

struct StructuralRiskAnalysis {
    double stability{};
    double protected_prefix_importance{};
    StuckStateClassification stuck;
};

[[nodiscard]] StructuralRiskAnalysis analyze_structural_risk(
    Board board, BoardCorner corner, SnakeOrientation orientation) {
    const auto path = get_snake_path(corner, orientation);
    const auto cells = decode(board);
    const auto maximum = *std::max_element(cells.begin(), cells.end());

    std::size_t anchored_length = 0;
    if (maximum != 0 && cells[path[0]] == maximum) {
        anchored_length = 1;
        while (anchored_length < kCellCount) {
            const auto previous = cells[path[anchored_length - 1]];
            if (previous <= 1 || cells[path[anchored_length]] + 1U != previous) {
                break;
            }
            ++anchored_length;
        }
    }
    std::size_t protected_length = 0;
    while (protected_length < anchored_length &&
           cells[path[protected_length]] >= kMinimumProtectedExponent) {
        ++protected_length;
    }

    StructuralRiskAnalysis analysis{};
    double vulnerable_importance = 0.0;
    for (std::size_t index = 0; index < kCellCount / 2; ++index) {
        const auto exponent = static_cast<double>(cells[path[index]]);
        if (cells[path[index]] >= kMinimumProtectedExponent) {
            vulnerable_importance += position_weight(index) * exponent * exponent;
        }
        if (index < protected_length) {
            analysis.protected_prefix_importance +=
                position_weight(index) * exponent * exponent;
        }
    }

    analysis.stuck.minimum_displacement = std::numeric_limits<double>::infinity();
    double preserved_weight = 0.0;
    double legal_weight = 0.0;
    for (std::size_t direction_index = 0; direction_index < kDirections.size();
         ++direction_index) {
        const auto direction = kDirections[direction_index];
        const auto moved = move(board, direction);
        if (!moved.moved) {
            continue;
        }
        const auto moved_cells = decode(moved.board);
        double displacement = 0.0;
        for (std::size_t index = 0; index < protected_length; ++index) {
            const auto exponent = cells[path[index]];
            if (moved_cells[path[index]] < exponent) {
                ++analysis.stuck.exposed_prefix_positions[direction_index];
                displacement += position_weight(index) *
                    static_cast<double>(exponent) * exponent;
            }
        }
        if (protected_length > 0 && moved_cells[path[0]] < cells[path[0]]) {
            const auto exponent = static_cast<double>(cells[path[0]]);
            displacement += 3.0 * exponent * exponent;
        }

        double vulnerability = 0.0;
        for (std::size_t index = 0; index < kCellCount / 2; ++index) {
            const auto exponent = cells[path[index]];
            if (exponent >= kMinimumProtectedExponent &&
                moved_cells[path[index]] < exponent) {
                vulnerability += position_weight(index) *
                    static_cast<double>(exponent) * exponent;
            }
        }

        analysis.stuck.legal_moves[direction_index] = true;
        analysis.stuck.merge_moves[direction_index] = moved.score > 0;
        analysis.stuck.merge_move_count += moved.score > 0 ? 1U : 0U;
        ++analysis.stuck.legal_move_count;
        analysis.stuck.minimum_displacement = std::min(
            analysis.stuck.minimum_displacement, displacement);
        if (displacement == 0.0) {
            analysis.stuck.safe_moves[direction_index] = true;
            ++analysis.stuck.safe_move_count;
        } else {
            analysis.stuck.escape_moves[direction_index] = true;
        }

        const auto direction_weight = preferred_direction(corner, direction) ? 1.0 : 0.25;
        legal_weight += direction_weight;
        preserved_weight += direction_weight / (1.0 + vulnerability);
    }

    if (analysis.stuck.legal_move_count == 0) {
        analysis.stuck.minimum_displacement = 0.0;
    }
    analysis.stuck.is_stuck = protected_length >= 2 &&
                              analysis.stuck.legal_move_count > 0 &&
                              analysis.stuck.safe_move_count == 0;
    analysis.stuck.is_adverse = analysis.stuck.is_stuck &&
                                analysis.stuck.minimum_displacement > 0.0;
    analysis.stability = legal_weight == 0.0
        ? 0.0
        : vulnerable_importance * preserved_weight / legal_weight;
    return analysis;
}

}  // namespace

SnakePath get_snake_path(BoardCorner corner, SnakeOrientation orientation) noexcept {
    SnakePath path{};
    const auto starting_row = is_bottom(corner) ? 3 : 0;
    const auto starting_column = is_right(corner) ? 3 : 0;
    const auto row_step = is_bottom(corner) ? -1 : 1;
    const auto column_step = is_right(corner) ? -1 : 1;

    std::size_t output = 0;
    for (int line = 0; line < 4; ++line) {
        for (int offset = 0; offset < 4; ++offset) {
            const auto alternating_offset = line % 2 == 0 ? offset : 3 - offset;
            int row = 0;
            int column = 0;
            if (orientation == SnakeOrientation::horizontal_first) {
                row = starting_row + row_step * line;
                column = starting_column + column_step * alternating_offset;
            } else {
                row = starting_row + row_step * alternating_offset;
                column = starting_column + column_step * line;
            }
            path[output++] = static_cast<std::size_t>(row * 4 + column);
        }
    }
    return path;
}

double main_line_score(
    Board board, BoardCorner corner, SnakeOrientation orientation) noexcept {
    const auto path = get_snake_path(corner, orientation);
    if (board.exponent_high_bits == 0) {
        const auto& tables = main_line_tables();
        double score = 0.0;
        std::array<std::uint8_t, kCellCount> path_exponents{};
        for (std::size_t block = 0; block < 4; ++block) {
            std::uint16_t row = 0;
            for (std::size_t offset = 0; offset < 4; ++offset) {
                const auto path_index = block * 4 + offset;
                const auto exponent = static_cast<std::uint8_t>(
                    (board.packed_exponents >> (path[path_index] * 4U)) & 0xFU);
                path_exponents[path_index] = exponent;
                row |= static_cast<std::uint16_t>(exponent) << (offset * 4U);
            }
            score += tables.score(block, row);
        }
        for (const auto index : {3U, 7U, 11U}) {
            score += adjacency_score(
                path_exponents[index], path_exponents[index + 1U], index);
        }
        return score;
    }

    const auto cells = decode(board);
    double score = 0.0;

    for (std::size_t index = 0; index < kCellCount; ++index) {
        const auto exponent = static_cast<double>(cells[path[index]]);
        score += position_weight(index) * exponent * exponent;
    }

    for (std::size_t index = 0; index + 1 < kCellCount; ++index) {
        const auto current = static_cast<int>(cells[path[index]]);
        const auto next = static_cast<int>(cells[path[index + 1]]);
        if (current == 0) {
            continue;
        }
        score += adjacency_score(current, next, index);
    }
    return score;
}

std::size_t anchored_main_line_length(
    Board board, BoardCorner corner, SnakeOrientation orientation) noexcept {
    const auto path = get_snake_path(corner, orientation);
    const auto cells = decode(board);
    const auto maximum = max_exponent(board);
    if (maximum == 0 || cells[path[0]] != maximum) {
        return 0;
    }

    std::size_t length = 1;
    while (length < kCellCount) {
        const auto previous = cells[path[length - 1]];
        if (previous <= 1 || cells[path[length]] + 1U != previous) {
            break;
        }
        ++length;
    }
    return length;
}

std::optional<MainLineFrontier> main_line_frontier(
    Board board, BoardCorner corner, SnakeOrientation orientation) noexcept {
    const auto path = get_snake_path(corner, orientation);
    const auto cells = decode(board);
    const auto length = anchored_main_line_length(board, corner, orientation);
    if (length == kCellCount) {
        return std::nullopt;
    }
    const auto desired = length == 0
        ? max_exponent(board)
        : static_cast<std::uint8_t>(std::max<int>(cells[path[length - 1]] - 1, 1));
    const auto cell_index = path[length];
    return MainLineFrontier{
        length,
        cell_index,
        cell_index / kBoardWidth,
        cell_index % kBoardWidth,
        cells[cell_index],
        desired,
    };
}

double structural_movement_penalty(
    Board old_board,
    Board new_board,
    BoardCorner corner,
    SnakeOrientation orientation) noexcept {
    const auto path = get_snake_path(corner, orientation);
    const auto old_cells = decode(old_board);
    const auto new_cells = decode(new_board);
    const auto length = protected_prefix_length(old_board, corner, orientation);
    double penalty = 0.0;
    for (std::size_t index = 0; index < length; ++index) {
        const auto old_exponent = old_cells[path[index]];
        if (new_cells[path[index]] < old_exponent) {
            const auto magnitude = static_cast<double>(old_exponent) * old_exponent;
            penalty += position_weight(index) * magnitude;
        }
    }
    if (length > 0 && new_cells[path[0]] < old_cells[path[0]]) {
        const auto maximum = static_cast<double>(old_cells[path[0]]);
        penalty += 3.0 * maximum * maximum;
    }
    return penalty;
}

double structural_stability_score(
    Board board, BoardCorner corner, SnakeOrientation orientation) {
    return analyze_structural_risk(board, corner, orientation).stability;
}

StuckStateClassification classify_stuck_state(
    Board board, BoardCorner corner, SnakeOrientation orientation) {
    return analyze_structural_risk(board, corner, orientation).stuck;
}

StructuralFeatures extract_structural_features(Board board, BoardCorner corner) {
    const auto selected = select_main_line(board, corner);
    StructuralFeatures features{};
    features.baseline = extract_baseline_features(board);
    features.main_line = selected.score;
    features.selected_orientation = selected.orientation;
    features.anchored_prefix_length = anchored_main_line_length(
        board, corner, selected.orientation);
    const auto risk = analyze_structural_risk(board, corner, selected.orientation);
    features.structural_stability = risk.stability;
    if (risk.stuck.is_adverse) {
        features.adverse_stuck_penalty = risk.stuck.minimum_displacement +
            0.25 * risk.protected_prefix_importance;
    }
    return features;
}

StructuralHeuristic::StructuralHeuristic(StructuralWeights weights, BoardCorner corner)
    : weights_(weights), corner_(corner) {}

double StructuralHeuristic::evaluate(Board board) const {
    const auto baseline = extract_baseline_features(board);
    const auto selected = select_main_line(board, corner_);
    auto value = weights_.baseline.empty_cells * baseline.empty_cells +
                 weights_.baseline.monotonicity * baseline.monotonicity +
                 weights_.baseline.smoothness * baseline.smoothness +
                 weights_.baseline.corner_preference * baseline.corner_preference +
                 weights_.main_line * selected.score;
    if (weights_.structural_stability != 0.0 || weights_.adverse_stuck != 0.0) {
        const auto risk = analyze_structural_risk(board, corner_, selected.orientation);
        value += weights_.structural_stability * risk.stability;
        if (risk.stuck.is_adverse) {
            const auto penalty = risk.stuck.minimum_displacement +
                0.25 * risk.protected_prefix_importance;
            value -= weights_.adverse_stuck * penalty;
        }
    }
    return value;
}

double StructuralHeuristic::evaluate_transition(Board old_board, Board new_board) const {
    if (weights_.structural_displacement == 0.0) {
        return 0.0;
    }
    const auto selected = select_main_line(old_board, corner_);
    return -weights_.structural_displacement * structural_movement_penalty(
        old_board, new_board, corner_, selected.orientation);
}

const StructuralWeights& StructuralHeuristic::weights() const noexcept {
    return weights_;
}

BoardCorner StructuralHeuristic::corner() const noexcept {
    return corner_;
}

StructuralWeights main_line_ablation_weights() noexcept {
    auto weights = StructuralWeights{};
    weights.structural_stability = 0.0;
    weights.adverse_stuck = 0.0;
    weights.structural_displacement = 0.0;
    return weights;
}

StructuralWeights movement_ablation_weights() noexcept {
    auto weights = StructuralWeights{};
    weights.structural_stability = 0.0;
    weights.adverse_stuck = 0.0;
    return weights;
}

}  // namespace adversarial_2048
