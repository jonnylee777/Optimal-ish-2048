#include "evaluation/baseline_heuristic.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>

namespace adversarial_2048 {
namespace {

[[nodiscard]] double line_monotonicity(
    const std::array<std::uint8_t, kBoardWidth>& line) noexcept {
    std::array<std::uint8_t, kBoardWidth> occupied{};
    std::size_t count = 0;
    for (const auto exponent : line) {
        if (exponent != 0) {
            occupied[count++] = exponent;
        }
    }

    double increasing_violations = 0.0;
    double decreasing_violations = 0.0;
    for (std::size_t index = 1; index < count; ++index) {
        const auto previous = static_cast<double>(occupied[index - 1]);
        const auto current = static_cast<double>(occupied[index]);
        increasing_violations += std::max(previous - current, 0.0);
        decreasing_violations += std::max(current - previous, 0.0);
    }
    return -std::min(increasing_violations, decreasing_violations);
}

struct RowFeatures {
    std::uint8_t empty_cells{};
    std::int16_t monotonicity{};
    std::int16_t smoothness{};
    std::uint8_t maximum{};
};

class BaselineFeatureTables {
public:
    static constexpr std::size_t kRowCount = 1U << 16U;

    BaselineFeatureTables() {
        for (std::size_t value = 0; value < kRowCount; ++value) {
            std::array<std::uint8_t, kBoardWidth> line{};
            for (std::size_t index = 0; index < kBoardWidth; ++index) {
                line[index] = static_cast<std::uint8_t>((value >> (index * 4U)) & 0xFU);
            }

            RowFeatures features{};
            features.monotonicity = static_cast<std::int16_t>(line_monotonicity(line));
            std::uint8_t previous_occupied = 0;
            for (const auto exponent : line) {
                if (exponent == 0) {
                    ++features.empty_cells;
                    continue;
                }
                features.maximum = std::max(features.maximum, exponent);
                if (previous_occupied != 0) {
                    features.smoothness -= static_cast<std::int16_t>(
                        std::abs(static_cast<int>(previous_occupied) -
                                 static_cast<int>(exponent)));
                }
                previous_occupied = exponent;
            }
            rows_[value] = features;
        }
    }

    [[nodiscard]] const RowFeatures& get(std::uint16_t row) const noexcept {
        return rows_[row];
    }

private:
    std::array<RowFeatures, kRowCount> rows_{};
};

[[nodiscard]] const BaselineFeatureTables& feature_tables() {
    static const BaselineFeatureTables tables;
    return tables;
}

[[nodiscard]] BaselineFeatures extract_extended_features(Board board) noexcept {
    const auto cells = decode(board);
    BaselineFeatures features{};
    features.empty_cells = static_cast<double>(empty_count(board));

    for (std::size_t line = 0; line < kBoardWidth; ++line) {
        std::array<std::uint8_t, kBoardWidth> row{};
        std::array<std::uint8_t, kBoardWidth> column{};
        for (std::size_t offset = 0; offset < kBoardWidth; ++offset) {
            row[offset] = cells[line * kBoardWidth + offset];
            column[offset] = cells[offset * kBoardWidth + line];
        }
        features.monotonicity += line_monotonicity(row);
        features.monotonicity += line_monotonicity(column);
    }

    for (std::size_t row = 0; row < kBoardWidth; ++row) {
        for (std::size_t column = 0; column < kBoardWidth; ++column) {
            const auto index = row * kBoardWidth + column;
            const auto exponent = cells[index];
            if (exponent == 0) {
                continue;
            }

            for (std::size_t next_column = column + 1;
                 next_column < kBoardWidth;
                 ++next_column) {
                const auto neighbor = cells[row * kBoardWidth + next_column];
                if (neighbor != 0) {
                    features.smoothness -= std::abs(
                        static_cast<double>(exponent) - static_cast<double>(neighbor));
                    break;
                }
            }
            for (std::size_t next_row = row + 1; next_row < kBoardWidth; ++next_row) {
                const auto neighbor = cells[next_row * kBoardWidth + column];
                if (neighbor != 0) {
                    features.smoothness -= std::abs(
                        static_cast<double>(exponent) - static_cast<double>(neighbor));
                    break;
                }
            }
        }
    }

    const auto maximum = max_exponent(board);
    if (maximum != 0 &&
        (cells[0] == maximum || cells[kBoardWidth - 1] == maximum ||
         cells[kCellCount - kBoardWidth] == maximum ||
         cells[kCellCount - 1] == maximum)) {
        features.corner_preference = static_cast<double>(maximum);
    }
    return features;
}

}  // namespace

BaselineFeatures extract_baseline_features(Board board) noexcept {
    if (board.exponent_high_bits != 0) {
        return extract_extended_features(board);
    }

    const auto& tables = feature_tables();
    const auto transposed = transpose(board).packed_exponents;
    BaselineFeatures features{};
    std::uint8_t maximum = 0;
    for (std::size_t row_index = 0; row_index < kBoardWidth; ++row_index) {
        const auto shift = row_index * 16U;
        const auto row = tables.get(static_cast<std::uint16_t>(
            (board.packed_exponents >> shift) & 0xFFFFU));
        const auto column = tables.get(static_cast<std::uint16_t>(
            (transposed >> shift) & 0xFFFFU));
        features.empty_cells += row.empty_cells;
        features.monotonicity += row.monotonicity + column.monotonicity;
        features.smoothness += row.smoothness + column.smoothness;
        maximum = std::max(maximum, row.maximum);
    }

    if (maximum != 0) {
        constexpr std::array<std::size_t, 4> corner_shifts{0, 12, 48, 60};
        for (const auto shift : corner_shifts) {
            if (((board.packed_exponents >> shift) & 0xFU) == maximum) {
                features.corner_preference = static_cast<double>(maximum);
                break;
            }
        }
    }
    return features;
}

BaselineHeuristic::BaselineHeuristic(BaselineWeights weights)
    : weights_(weights) {}

double BaselineHeuristic::evaluate(Board board) const {
    const auto features = extract_baseline_features(board);
    return weights_.empty_cells * features.empty_cells +
           weights_.monotonicity * features.monotonicity +
           weights_.smoothness * features.smoothness +
           weights_.corner_preference * features.corner_preference;
}

const BaselineWeights& BaselineHeuristic::weights() const noexcept {
    return weights_;
}

}  // namespace adversarial_2048
