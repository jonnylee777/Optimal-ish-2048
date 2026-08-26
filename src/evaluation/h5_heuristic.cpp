#include "evaluation/h5_heuristic.hpp"

#include "core/board.hpp"

#include <algorithm>
#include <array>

namespace adversarial_2048 {
namespace {

// Reference's TILE_WEIGHT_MAP (native_core/src/AIPlayer.cpp). Index is the
// tile's exponent (0 = empty, 1 = tile 2, ...). Deliberately saturating:
// weight growth stops well before the table's own 4-bit ceiling.
constexpr std::array<std::int32_t, 16> kTileWeight{
    0, 2, 4, 8, 16, 32, 64, 128, 248, 388, 488, 518, 519, 519, 519, 520};

[[nodiscard]] std::int32_t weight_of(std::uint8_t exponent) noexcept {
    return exponent < kTileWeight.size() ? kTileWeight[exponent] : kTileWeight.back();
}

struct RowScorePair {
    std::int16_t forward{};  // score reading the line left-to-right (or top-to-bottom)
    std::int16_t reverse{};  // score reading the same line the other way
};

class H5RowTable {
public:
    static constexpr std::size_t kRowCount = 1U << 16U;

    H5RowTable() {
        for (std::size_t value = 0; value < kRowCount; ++value) {
            std::array<std::int32_t, 4> forward_weights{};
            std::array<std::int32_t, 4> reverse_weights{};
            for (std::size_t index = 0; index < 4; ++index) {
                const auto exponent =
                    static_cast<std::uint8_t>((value >> (index * 4U)) & 0xFU);
                const auto weight = weight_of(exponent);
                forward_weights[index] = weight;
                reverse_weights[3 - index] = weight;
            }
            rows_[value] = RowScorePair{
                static_cast<std::int16_t>(h5_line_score(forward_weights)),
                static_cast<std::int16_t>(h5_line_score(reverse_weights)),
            };
        }
    }

    [[nodiscard]] const RowScorePair& get(std::uint16_t row) const noexcept {
        return rows_[row];
    }

private:
    std::array<RowScorePair, kRowCount> rows_{};
};

[[nodiscard]] const H5RowTable& row_table() {
    static const H5RowTable table;
    return table;
}

// Exact fallback for boards with exponents above 15 (outside the 4-bit row
// table; the reference has no such boards at all). Recomputes both readings
// of every row and column directly instead of going through the table.
[[nodiscard]] double evaluate_extended(Board board) noexcept {
    const auto cells = decode(board);
    std::int64_t sum_x1 = 0;
    std::int64_t sum_x2 = 0;
    std::int64_t sum_y1 = 0;
    std::int64_t sum_y2 = 0;

    for (std::size_t line = 0; line < kBoardWidth; ++line) {
        std::array<std::int32_t, 4> row_forward{};
        std::array<std::int32_t, 4> row_reverse{};
        std::array<std::int32_t, 4> column_forward{};
        std::array<std::int32_t, 4> column_reverse{};
        for (std::size_t offset = 0; offset < kBoardWidth; ++offset) {
            const auto row_weight = weight_of(cells[line * kBoardWidth + offset]);
            const auto column_weight = weight_of(cells[offset * kBoardWidth + line]);
            row_forward[offset] = row_weight;
            row_reverse[3 - offset] = row_weight;
            column_forward[offset] = column_weight;
            column_reverse[3 - offset] = column_weight;
        }
        // Truncate to int16_t per line, exactly like the reference's
        // ScorePair.d1/d2 and this file's own RowScorePair — the sum must be
        // built from already-truncated per-line values, not truncated after
        // summing, or this path would silently disagree with the table path.
        sum_x1 += static_cast<std::int16_t>(h5_line_score(row_forward));
        sum_x2 += static_cast<std::int16_t>(h5_line_score(row_reverse));
        sum_y1 += static_cast<std::int16_t>(h5_line_score(column_forward));
        sum_y2 += static_cast<std::int16_t>(h5_line_score(column_reverse));
    }
    return static_cast<double>(std::max(sum_x1, sum_x2) + std::max(sum_y1, sum_y2));
}

}  // namespace

std::int32_t h5_tile_weight(std::uint8_t exponent) noexcept {
    return weight_of(exponent);
}

std::int32_t h5_line_score(const std::array<std::int32_t, 4>& weights) noexcept {
    // --- "dpdf": rewards a monotonic run, with a steep penalty for breaking
    // one once tiles are large. Ported exactly, including its integer
    // truncation, from diffs_evaluation_func in the reference.
    std::int32_t score_dpdf = weights[0];
    for (std::size_t x = 0; x < 3; ++x) {
        if (weights[x] < weights[x + 1]) {
            if (weights[x] > 400) {
                score_dpdf += (weights[x] << 1) +
                              (weights[x + 1] - weights[x]) * static_cast<std::int32_t>(x);
            } else if (weights[x] > 300 && x == 1 && weights[0] > weights[1]) {
                score_dpdf += (weights[x] << 1);
            } else {
                score_dpdf -= (weights[x + 1] - weights[x]) << 3;
                score_dpdf -= weights[x + 1] * 3;
                if (x < 2 && weights[x + 2] < weights[x + 1] && weights[x + 1] > 30) {
                    score_dpdf -= std::max(80, weights[x + 1]);
                }
            }
        } else if (x < 2) {
            score_dpdf += weights[x + 1] + weights[x];
        } else {
            score_dpdf += static_cast<std::int32_t>(
                (static_cast<double>(weights[x + 1]) + static_cast<double>(weights[x])) * 0.5);
        }
    }
    if (weights[0] > 400 && weights[1] > 300 && weights[2] > 200 && weights[2] > weights[3] &&
        weights[3] < 300) {
        score_dpdf += weights[3] >> 2;
    }

    // --- "t": rewards large tiles walling both ends of the line (the T
    // formation's column-0 domino, read as a line) with useful tiles between.
    std::int32_t score_t{};
    const auto min_ends = std::min(weights[0], weights[3]);
    if (min_ends < 32) {
        score_t = -16384;  // both ends nearly empty: this line is dead weight
    } else if ((weights[0] < weights[1] && weights[0] < 400) ||
               (weights[3] < weights[2] && weights[3] < 400)) {
        score_t = -(std::max(weights[1], weights[2]) * 10);
    } else {
        score_t = static_cast<std::int32_t>(
            (static_cast<double>(weights[0]) * 1.8 + static_cast<double>(weights[3]) * 1.8) +
            static_cast<double>(std::max(weights[1], weights[2])) * 1.5 +
            static_cast<double>(std::min(std::int32_t{160}, std::min(weights[1], weights[2]))) *
                2.5);
        if (std::min(weights[1], weights[2]) < 8) {
            score_t -= 60;
        }
    }

    int zero_count = 0;
    for (const auto weight : weights) {
        if (weight == 0) {
            ++zero_count;
        }
    }
    const auto sum_last_three = weights[1] + weights[2] + weights[3];
    std::int32_t penalty = 0;
    if (weights[0] > 100 && ((zero_count > 1 && sum_last_three < 32) || sum_last_three < 12)) {
        penalty = 4;
    }

    return std::max(score_dpdf, score_t) / 4 - penalty;
}

double H5Heuristic::evaluate(Board board) const {
    if (board.exponent_high_bits != 0) {
        return evaluate_extended(board);
    }

    const auto& table = row_table();
    const auto transposed = transpose(board).packed_exponents;
    std::int64_t sum_x1 = 0;
    std::int64_t sum_x2 = 0;
    std::int64_t sum_y1 = 0;
    std::int64_t sum_y2 = 0;

    for (std::size_t index = 0; index < kBoardWidth; ++index) {
        const auto shift = index * 16U;
        const auto row = table.get(
            static_cast<std::uint16_t>((board.packed_exponents >> shift) & 0xFFFFU));
        sum_x1 += row.forward;
        sum_x2 += row.reverse;

        const auto column = table.get(
            static_cast<std::uint16_t>((transposed >> shift) & 0xFFFFU));
        sum_y1 += column.forward;
        sum_y2 += column.reverse;
    }
    return static_cast<double>(std::max(sum_x1, sum_x2) + std::max(sum_y1, sum_y2));
}

}  // namespace adversarial_2048
