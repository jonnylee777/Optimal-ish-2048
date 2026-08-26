#pragma once

#include "core/board.hpp"

#include <cstdint>

namespace adversarial_2048::tablebase {

// Wall-aware moves for the board-size variants. Ported from the reference's
// `VBoardMover` / `WallMergePolicy` (native_core/include/VBoardMover.h):
//
//   * A 0xF nibble is a WALL: immovable, and it splits its line into
//     independent segments that compact separately.
//   * 0xE (16384) is the largest mergeable tile — merging two of them would
//     produce 0xF, which is reserved as the wall marker, so it is forbidden.
//
// Operates on the raw packed representation (no extension plane); variants
// never produce tiles above 0xE by construction.
struct VariantMoveResult {
    std::uint64_t packed{};
    std::uint64_t score{};
    bool moved{};
};

[[nodiscard]] VariantMoveResult move_variant(
    std::uint64_t packed, Direction direction) noexcept;

[[nodiscard]] bool variant_has_move(std::uint64_t packed) noexcept;

}  // namespace adversarial_2048::tablebase
