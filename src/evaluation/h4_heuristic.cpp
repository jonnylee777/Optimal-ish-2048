#include "evaluation/h4_heuristic.hpp"

#include "core/board.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>

namespace adversarial_2048 {
namespace {

constexpr double kMonotonicityPower = 4.0;
constexpr double kSumPower = 3.5;

// Computes the four H4 features for one line of exponents. Shared by the
// 65536-entry table builder (ordinary boards) and the exact fallback used
// for extended boards, so both paths are guaranteed identical by
// construction rather than by parallel maintenance.
[[nodiscard]] H4Features line_features(
    const std::array<std::uint8_t, kBoardWidth>& line) noexcept {
    H4Features features{};

    int previous = 0;
    int run = 0;
    for (const auto rank : line) {
        features.sum += std::pow(static_cast<double>(rank), kSumPower);
        if (rank == 0) {
            features.empty += 1.0;
            continue;
        }
        if (previous == static_cast<int>(rank)) {
            ++run;
        } else if (run > 0) {
            features.merges += 1.0 + static_cast<double>(run);
            run = 0;
        }
        previous = static_cast<int>(rank);
    }
    if (run > 0) {
        features.merges += 1.0 + static_cast<double>(run);
    }

    double descending_left = 0.0;
    double descending_right = 0.0;
    for (std::size_t index = 1; index < kBoardWidth; ++index) {
        const auto previous_ranked = std::pow(
            static_cast<double>(line[index - 1]), kMonotonicityPower);
        const auto current_ranked = std::pow(
            static_cast<double>(line[index]), kMonotonicityPower);
        if (line[index - 1] > line[index]) {
            descending_left += previous_ranked - current_ranked;
        } else {
            descending_right += current_ranked - previous_ranked;
        }
    }
    features.monotonicity = std::min(descending_left, descending_right);
    return features;
}

class H4FeatureTables {
public:
    static constexpr std::size_t kRowCount = 1U << 16U;

    H4FeatureTables() {
        for (std::size_t value = 0; value < kRowCount; ++value) {
            std::array<std::uint8_t, kBoardWidth> line{};
            for (std::size_t index = 0; index < kBoardWidth; ++index) {
                line[index] = static_cast<std::uint8_t>((value >> (index * 4U)) & 0xFU);
            }
            rows_[value] = line_features(line);
        }
    }

    [[nodiscard]] const H4Features& get(std::uint16_t row) const noexcept {
        return rows_[row];
    }

private:
    std::array<H4Features, kRowCount> rows_{};
};

[[nodiscard]] const H4FeatureTables& feature_tables() {
    static const H4FeatureTables tables;
    return tables;
}

void accumulate(H4Features& total, const H4Features& part) noexcept {
    total.empty += part.empty;
    total.merges += part.merges;
    total.monotonicity += part.monotonicity;
    total.sum += part.sum;
}

// Exact path for boards containing exponents above 15, which fall outside
// the four-bit row tables. Mirrors extract_baseline_features' same split.
[[nodiscard]] H4Features extract_extended_features(Board board) noexcept {
    const auto cells = decode(board);
    H4Features total{};
    for (std::size_t line = 0; line < kBoardWidth; ++line) {
        std::array<std::uint8_t, kBoardWidth> row{};
        std::array<std::uint8_t, kBoardWidth> column{};
        for (std::size_t offset = 0; offset < kBoardWidth; ++offset) {
            row[offset] = cells[line * kBoardWidth + offset];
            column[offset] = cells[offset * kBoardWidth + line];
        }
        accumulate(total, line_features(row));
        accumulate(total, line_features(column));
    }
    return total;
}

}  // namespace

H4Features extract_h4_features(Board board) noexcept {
    if (board.exponent_high_bits != 0) {
        return extract_extended_features(board);
    }

    const auto& tables = feature_tables();
    const auto transposed = transpose(board).packed_exponents;
    H4Features total{};
    for (std::size_t row_index = 0; row_index < kBoardWidth; ++row_index) {
        const auto shift = row_index * 16U;
        accumulate(total, tables.get(static_cast<std::uint16_t>(
            (board.packed_exponents >> shift) & 0xFFFFU)));
        accumulate(total, tables.get(static_cast<std::uint16_t>(
            (transposed >> shift) & 0xFFFFU)));
    }
    return total;
}

H4Heuristic::H4Heuristic(H4Weights weights) : weights_(weights) {}

double H4Heuristic::evaluate(Board board) const {
    const auto features = extract_h4_features(board);
    return kH4LostPenaltyTotal +
           weights_.empty * features.empty +
           weights_.merges * features.merges -
           weights_.monotonicity * features.monotonicity -
           weights_.sum * features.sum;
}

const H4Weights& H4Heuristic::weights() const noexcept {
    return weights_;
}

}  // namespace adversarial_2048
