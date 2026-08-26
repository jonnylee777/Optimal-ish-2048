#pragma once

#include "evaluation/evaluator.hpp"

#include <array>
#include <cstdint>

namespace adversarial_2048 {

// H5: a faithful port of the reference project's SEARCH evaluator — not its
// tablebase — from game-difficulty/2048EndgameTablebase
// (`native_core/src/AIPlayer.cpp`: `diffs_evaluation_func`,
// `init_evaluate_tables`, `AIPlayer::evaluate`). See docs/tablebase.md.
//
// Two things make this different from H0-H4:
//
//  1. Saturating tile weights (`kTileWeight`): value plateaus at 520 past
//     exponent ~11, so a huge tile stops contributing extra magnitude and
//     only its position/structure matters.
//  2. Two competing per-line scores computed from the SAME weighted line —
//     a DPDF-style (monotonic run) score and a "T-formation" score that
//     rewards large tiles walling both ends of a line with useful tiles in
//     between — combined by reading each of the 4 rows and 4 columns in
//     BOTH directions, summing each direction's total separately, and only
//     then taking max(forward-total, reverse-total) per axis. Taking the
//     max after summing (not per line) is what makes the whole-board score
//     invariant under all 8 board symmetries even though a single line's
//     score is direction-dependent — see is_rotation_invariant().
//
// No tunable weights: every constant here is the reference's own hardcoded,
// already-tuned value, not something this port re-optimizes.
struct H5Heuristic final : Evaluator {
    [[nodiscard]] double evaluate(Board board) const override;
    [[nodiscard]] bool is_rotation_invariant() const noexcept override {
        return true;
    }
};

// Saturating exponent -> tile-weight table (reference's TILE_WEIGHT_MAP).
// Exposed for testing. Exponents 16-31 (our extension plane; the reference
// has no equivalent) continue the plateau at the table's own final value,
// consistent with the table's own design intent of "big tile, magnitude
// stops mattering."
[[nodiscard]] std::int32_t h5_tile_weight(std::uint8_t exponent) noexcept;

// The reference's `diffs_evaluation_func`, operating on one line already
// converted to tile weights (not raw exponents). Exposed for testing against
// an independent transcription.
[[nodiscard]] std::int32_t h5_line_score(const std::array<std::int32_t, 4>& weights) noexcept;

}  // namespace adversarial_2048
