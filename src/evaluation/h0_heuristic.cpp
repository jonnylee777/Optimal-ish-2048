#include "evaluation/h0_heuristic.hpp"

#include "core/board.hpp"

#include <algorithm>
#include <array>

namespace adversarial_2048 {
namespace {

struct RowFeatures {
    std::uint8_t empty_cells{};
    std::uint8_t maximum{};
};

class H0FeatureTables {
public:
    static constexpr std::size_t kRowCount = 1U << 16U;

    H0FeatureTables() {
        for (std::size_t value = 0; value < kRowCount; ++value) {
            RowFeatures features{};
            for (std::size_t index = 0; index < kBoardWidth; ++index) {
                const auto exponent =
                    static_cast<std::uint8_t>((value >> (index * 4U)) & 0xFU);
                if (exponent == 0) {
                    ++features.empty_cells;
                } else {
                    features.maximum = std::max(features.maximum, exponent);
                }
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

[[nodiscard]] const H0FeatureTables& feature_tables() {
    static const H0FeatureTables tables;
    return tables;
}

// The 12 of 16 cells touching the board edge, as nibble bit-shifts into
// packed_exponents: all of row 0 and row 3, plus the leftmost/rightmost
// cell of rows 1 and 2. Direct extraction (not a table) because there's no
// per-row decomposition that also tells us which cell holds the *board's*
// maximum, not just each row's own — this mirrors how corner_preference is
// computed in baseline_heuristic.cpp.
constexpr std::array<unsigned, 12> kEdgeShifts{
    0, 4, 8, 12,     // row 0
    16, 28,          // row 1: leftmost, rightmost
    32, 44,          // row 2: leftmost, rightmost
    48, 52, 56, 60,  // row 3
};

[[nodiscard]] H0Features extract_ordinary_features(Board board) noexcept {
    const auto& tables = feature_tables();
    H0Features features{};
    std::uint8_t maximum = 0;
    for (std::size_t row_index = 0; row_index < kBoardWidth; ++row_index) {
        const auto row = tables.get(static_cast<std::uint16_t>(
            (board.packed_exponents >> (row_index * 16U)) & 0xFFFFU));
        features.empty_cells += row.empty_cells;
        maximum = std::max(maximum, row.maximum);
    }
    if (maximum != 0) {
        for (const auto shift : kEdgeShifts) {
            if (((board.packed_exponents >> shift) & 0xFU) == maximum) {
                features.edge_max_bonus = static_cast<double>(maximum);
                break;
            }
        }
    }
    return features;
}

// Exact fallback for boards with exponents above 15 (outside the four-bit
// row tables) — same split as baseline_heuristic.cpp's extended path.
[[nodiscard]] H0Features extract_extended_features(Board board) noexcept {
    const auto cells = decode(board);
    H0Features features{};
    std::uint8_t maximum = 0;
    for (const auto exponent : cells) {
        if (exponent == 0) {
            features.empty_cells += 1.0;
        } else {
            maximum = std::max(maximum, exponent);
        }
    }
    if (maximum != 0) {
        constexpr std::array<std::size_t, 12> edge_indices{
            0, 1, 2, 3, 4, 7, 8, 11, 12, 13, 14, 15};
        for (const auto index : edge_indices) {
            if (cells[index] == maximum) {
                features.edge_max_bonus = static_cast<double>(maximum);
                break;
            }
        }
    }
    return features;
}

}  // namespace

H0Features extract_h0_features(Board board) noexcept {
    return board.exponent_high_bits != 0
        ? extract_extended_features(board)
        : extract_ordinary_features(board);
}

H0Heuristic::H0Heuristic(H0Weights weights) : weights_(weights) {}

double H0Heuristic::evaluate(Board board) const {
    const auto features = extract_h0_features(board);
    return weights_.empty_cells * features.empty_cells +
           weights_.edge_max_bonus * features.edge_max_bonus;
}

const H0Weights& H0Heuristic::weights() const noexcept {
    return weights_;
}

}  // namespace adversarial_2048
