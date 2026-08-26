#pragma once

#include "core/board.hpp"

#include <cstdint>
#include <string>
#include <vector>

namespace adversarial_2048::tablebase {

// A "formation" is an endgame shape: a set of cells locked to immovable large
// tiles, leaving the rest free. Ported from game-difficulty/2048EndgameTablebase
// (`native_core/include/Formation.h`, `docs_and_configs/patterns_config.json`).
//
// Convention, matching the reference exactly: nibble value 0xF (= kWallExponent)
// marks a locked/immovable large tile in a mask, and also acts as a wall in the
// board-size variants (2x4/3x3/3x4). A mask nibble of 0 means "don't care".
inline constexpr std::uint8_t kWallExponent = 0xF;

// The 8 elements of the dihedral group of the square. Index order matches the
// reference's `apply_sym_like` (`native_core/include/SymmetryUtils.h`) so ported
// data stays interpretable.
enum class Symmetry : std::uint8_t {
    identity = 0,
    reverse_lr = 1,
    reverse_ud = 2,
    transpose_main = 3,   // reflect across the main diagonal
    transpose_anti = 4,   // reflect across the anti-diagonal
    rotate_180 = 5,
    rotate_left = 6,
    rotate_right = 7,
};

// Which subgroup a formation may be quotiented by when storing states. Only
// symmetries that map the formation's own mask set and success cells onto
// themselves are valid, so this is a per-formation property.
enum class SymmetryMode : std::uint8_t {
    identity,    // no reduction
    full,        // all 8 dihedral elements
    diagonal,    // {identity, transpose_main}
    horizontal,  // {identity, reverse_lr}
    min24,       // 2x4 variant: rows 0 and 3 are walls
    min33,       // 3x3 variant: row 3 and column 3 are walls
    min34,       // 3x4 variant: row 3 is a wall
};

struct Formation {
    std::string name;
    // A board is "in" the formation if it matches ANY mask (they are
    // alternatives, e.g. the T formation's sliding domino has 3 positions).
    // An empty list means every board matches, used by the free/variant tables.
    std::vector<std::uint64_t> masks;
    // Nibble shifts (multiples of 4) at which a tile equal to the target
    // exponent counts as success. Empty means every board is a success.
    std::vector<std::uint8_t> success_shifts;
    SymmetryMode symmetry{SymmetryMode::identity};
    // Exponent of the tile we are trying to build, e.g. 9 for the 512 tile.
    std::uint8_t target_exponent{};
    // Boards the forward generation starts from.
    std::vector<std::uint64_t> seeds;
    // True for the board-size variants, which need the wall-aware mover.
    bool is_variant{};
};

[[nodiscard]] std::uint64_t apply_symmetry(std::uint64_t packed, Symmetry symmetry) noexcept;

// Returns the lexicographically smallest board in the orbit of `packed` under
// the subgroup selected by `mode`. Storing only this representative is what
// shrinks the table (up to 8x for `full`).
[[nodiscard]] std::uint64_t canonicalize(std::uint64_t packed, SymmetryMode mode) noexcept;

// True when every 0xF nibble of some mask is also 0xF in the board.
[[nodiscard]] bool matches_formation(
    std::uint64_t packed, const std::vector<std::uint64_t>& masks) noexcept;

[[nodiscard]] bool is_success(
    std::uint64_t packed,
    std::uint8_t target_exponent,
    const std::vector<std::uint8_t>& success_shifts) noexcept;

// Sum of tile face values (2^exponent), skipping 0xF walls/locked tiles. This
// is what layers are indexed by: a spawn adds 2 or 4 and merges preserve the
// sum, so the layer graph is acyclic.
[[nodiscard]] std::uint32_t free_tile_sum(std::uint64_t packed) noexcept;

// Built-in formations from the reference's patterns_config.json.
[[nodiscard]] Formation variant_2x4(std::uint8_t target_exponent);
[[nodiscard]] Formation variant_3x3(std::uint8_t target_exponent);
[[nodiscard]] Formation variant_3x4(std::uint8_t target_exponent);

}  // namespace adversarial_2048::tablebase
