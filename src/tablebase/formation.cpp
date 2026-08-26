#include "tablebase/formation.hpp"

#include <algorithm>
#include <array>

namespace adversarial_2048::tablebase {
namespace {

// Our packed layout puts cell (r,c) at nibble r*4+c, i.e. shift 4*(r*4+c), so
// (0,0) is the LOW nibble. The reference project uses the opposite order —
// (0,0) is its HIGH nibble. Rather than hand-converting its hex masks (a silent
// bit-order slip would be very hard to spot), formations here are built from
// explicit (row, column) lists.
[[nodiscard]] constexpr unsigned shift_of(std::size_t row, std::size_t column) noexcept {
    return static_cast<unsigned>(4U * (row * kBoardWidth + column));
}

using CellMap = std::array<std::uint8_t, kCellCount>;

// Builds a permutation where destination cell `d` takes its value from the
// source cell produced by `f(row, column)`.
template <typename Mapper>
[[nodiscard]] constexpr CellMap build_map(Mapper mapper) noexcept {
    CellMap map{};
    for (std::size_t row = 0; row < kBoardWidth; ++row) {
        for (std::size_t column = 0; column < kBoardWidth; ++column) {
            const auto [source_row, source_column] = mapper(row, column);
            map[row * kBoardWidth + column] =
                static_cast<std::uint8_t>(source_row * kBoardWidth + source_column);
        }
    }
    return map;
}

[[nodiscard]] std::uint64_t permute(std::uint64_t packed, const CellMap& map) noexcept {
    std::uint64_t result = 0;
    for (std::size_t destination = 0; destination < kCellCount; ++destination) {
        const auto nibble = (packed >> (4U * map[destination])) & 0xFULL;
        result |= nibble << (4U * destination);
    }
    return result;
}

using Cell = std::pair<std::size_t, std::size_t>;

// --- full-board dihedral group ---
constexpr auto kReverseLr = build_map(
    [](std::size_t r, std::size_t c) -> Cell { return {r, kBoardWidth - 1 - c}; });
constexpr auto kReverseUd = build_map(
    [](std::size_t r, std::size_t c) -> Cell { return {kBoardWidth - 1 - r, c}; });
constexpr auto kTransposeMain = build_map(
    [](std::size_t r, std::size_t c) -> Cell { return {c, r}; });
constexpr auto kTransposeAnti = build_map(
    [](std::size_t r, std::size_t c) -> Cell {
        return {kBoardWidth - 1 - c, kBoardWidth - 1 - r};
    });
constexpr auto kRotate180 = build_map(
    [](std::size_t r, std::size_t c) -> Cell {
        return {kBoardWidth - 1 - r, kBoardWidth - 1 - c};
    });
// Counter-clockwise: destination (r,c) reads source (c, 3-r).
constexpr auto kRotateLeft = build_map(
    [](std::size_t r, std::size_t c) -> Cell { return {c, kBoardWidth - 1 - r}; });
// Clockwise: destination (r,c) reads source (3-c, r).
constexpr auto kRotateRight = build_map(
    [](std::size_t r, std::size_t c) -> Cell { return {kBoardWidth - 1 - c, r}; });

// --- 3x3 sub-board group (row 3 and column 3 are walls and stay put) ---
template <typename Mapper>
[[nodiscard]] constexpr CellMap build_map_33(Mapper mapper) noexcept {
    return build_map([mapper](std::size_t r, std::size_t c) -> Cell {
        if (r >= 3 || c >= 3) {
            return {r, c};
        }
        return mapper(r, c);
    });
}

constexpr auto kExchangeCol02 = build_map_33(
    [](std::size_t r, std::size_t c) -> Cell { return {r, 2 - c}; });
constexpr auto kExchangeRow02 = build_map_33(
    [](std::size_t r, std::size_t c) -> Cell { return {2 - r, c}; });
constexpr auto kTranspose33 = build_map_33(
    [](std::size_t r, std::size_t c) -> Cell { return {c, r}; });
constexpr auto kAntiTranspose33 = build_map_33(
    [](std::size_t r, std::size_t c) -> Cell { return {2 - c, 2 - r}; });
constexpr auto kRotate180_33 = build_map_33(
    [](std::size_t r, std::size_t c) -> Cell { return {2 - r, 2 - c}; });
constexpr auto kRotateRight33 = build_map_33(
    [](std::size_t r, std::size_t c) -> Cell { return {2 - c, r}; });
constexpr auto kRotateLeft33 = build_map_33(
    [](std::size_t r, std::size_t c) -> Cell { return {c, 2 - r}; });

// --- 3x4 sub-board group (row 3 is a wall and stays put) ---
constexpr auto kReverseUd34 = build_map([](std::size_t r, std::size_t c) -> Cell {
    return r >= 3 ? Cell{r, c} : Cell{2 - r, c};
});
// Must equal reverse_lr composed with reverse_ud34, or {identity, reverse_lr,
// reverse_ud34, rotate_180_34} is not closed and canonicalize() stops being
// idempotent. Note row 3 gets its columns flipped too: harmless for real 3x4
// boards (row 3 is entirely walls, so flipping it is a no-op) and required for
// the group property to hold generally.
constexpr auto kRotate180_34 = build_map([](std::size_t r, std::size_t c) -> Cell {
    return r >= 3 ? Cell{r, kBoardWidth - 1 - c} : Cell{2 - r, kBoardWidth - 1 - c};
});

}  // namespace

std::uint64_t apply_symmetry(std::uint64_t packed, Symmetry symmetry) noexcept {
    switch (symmetry) {
        case Symmetry::identity: return packed;
        case Symmetry::reverse_lr: return permute(packed, kReverseLr);
        case Symmetry::reverse_ud: return permute(packed, kReverseUd);
        case Symmetry::transpose_main: return permute(packed, kTransposeMain);
        case Symmetry::transpose_anti: return permute(packed, kTransposeAnti);
        case Symmetry::rotate_180: return permute(packed, kRotate180);
        case Symmetry::rotate_left: return permute(packed, kRotateLeft);
        case Symmetry::rotate_right: return permute(packed, kRotateRight);
    }
    return packed;
}

std::uint64_t canonicalize(std::uint64_t packed, SymmetryMode mode) noexcept {
    switch (mode) {
        case SymmetryMode::identity:
            return packed;
        case SymmetryMode::full: {
            auto best = packed;
            for (const auto& map : {kReverseLr, kReverseUd, kTransposeMain, kTransposeAnti,
                                    kRotate180, kRotateLeft, kRotateRight}) {
                best = std::min(best, permute(packed, map));
            }
            return best;
        }
        case SymmetryMode::diagonal:
            return std::min(packed, permute(packed, kTransposeMain));
        case SymmetryMode::horizontal:
            return std::min(packed, permute(packed, kReverseLr));
        case SymmetryMode::min24: {
            auto best = packed;
            for (const auto& map : {kReverseLr, kReverseUd, kRotate180}) {
                best = std::min(best, permute(packed, map));
            }
            return best;
        }
        case SymmetryMode::min33: {
            auto best = packed;
            for (const auto& map : {kExchangeCol02, kExchangeRow02, kTranspose33,
                                    kAntiTranspose33, kRotate180_33, kRotateRight33,
                                    kRotateLeft33}) {
                best = std::min(best, permute(packed, map));
            }
            return best;
        }
        case SymmetryMode::min34: {
            auto best = packed;
            for (const auto& map : {kReverseLr, kReverseUd34, kRotate180_34}) {
                best = std::min(best, permute(packed, map));
            }
            return best;
        }
    }
    return packed;
}

bool matches_formation(
    std::uint64_t packed, const std::vector<std::uint64_t>& masks) noexcept {
    if (masks.empty()) {
        return true;
    }
    for (const auto mask : masks) {
        if ((packed & mask) == mask) {
            return true;
        }
    }
    return false;
}

bool is_success(
    std::uint64_t packed,
    std::uint8_t target_exponent,
    const std::vector<std::uint8_t>& success_shifts) noexcept {
    if (success_shifts.empty()) {
        return true;
    }
    for (const auto shift : success_shifts) {
        if (((packed >> shift) & 0xFULL) == static_cast<std::uint64_t>(target_exponent)) {
            return true;
        }
    }
    return false;
}

std::uint32_t free_tile_sum(std::uint64_t packed) noexcept {
    std::uint32_t total = 0;
    for (std::size_t index = 0; index < kCellCount; ++index) {
        const auto exponent = static_cast<std::uint8_t>((packed >> (4U * index)) & 0xFULL);
        if (exponent != 0 && exponent != kWallExponent) {
            total += 1U << exponent;
        }
    }
    return total;
}

namespace {

// Every cell of `rows` x `columns` outside the playable rectangle becomes a wall.
[[nodiscard]] std::uint64_t wall_outside(std::size_t first_row, std::size_t last_row,
                                         std::size_t first_column, std::size_t last_column) {
    std::uint64_t mask = 0;
    for (std::size_t row = 0; row < kBoardWidth; ++row) {
        for (std::size_t column = 0; column < kBoardWidth; ++column) {
            const bool playable = row >= first_row && row <= last_row &&
                                  column >= first_column && column <= last_column;
            if (!playable) {
                mask |= static_cast<std::uint64_t>(kWallExponent) << shift_of(row, column);
            }
        }
    }
    return mask;
}

[[nodiscard]] std::vector<std::uint8_t> shifts_inside(
    std::size_t first_row, std::size_t last_row,
    std::size_t first_column, std::size_t last_column) {
    std::vector<std::uint8_t> shifts;
    for (std::size_t row = first_row; row <= last_row; ++row) {
        for (std::size_t column = first_column; column <= last_column; ++column) {
            shifts.push_back(static_cast<std::uint8_t>(shift_of(row, column)));
        }
    }
    return shifts;
}

}  // namespace

// The three board-size variants from patterns_config.json. Each walls off part
// of the 4x4 grid; `valid pattern` is empty there, so the wall (carried in the
// seed and preserved by the wall-aware mover) is the only structural constraint.
Formation variant_2x4(std::uint8_t target_exponent) {
    // Rows 0 and 3 walled; 8 playable cells in rows 1-2.
    return Formation{
        "2x4", {}, shifts_inside(1, 2, 0, kBoardWidth - 1),
        SymmetryMode::min24, target_exponent, {wall_outside(1, 2, 0, kBoardWidth - 1)}, true,
    };
}

Formation variant_3x3(std::uint8_t target_exponent) {
    // Row 3 and column 3 walled; 9 playable cells in rows 0-2, columns 0-2.
    return Formation{
        "3x3", {}, shifts_inside(0, 2, 0, 2),
        SymmetryMode::min33, target_exponent, {wall_outside(0, 2, 0, 2)}, true,
    };
}

Formation variant_3x4(std::uint8_t target_exponent) {
    // Row 3 walled; 12 playable cells in rows 0-2.
    return Formation{
        "3x4", {}, shifts_inside(0, 2, 0, kBoardWidth - 1),
        SymmetryMode::min34, target_exponent, {wall_outside(0, 2, 0, kBoardWidth - 1)}, true,
    };
}

}  // namespace adversarial_2048::tablebase
