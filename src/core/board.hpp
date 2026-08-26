#pragma once

#include <array>
#include <cstddef>
#include <cstdint>

namespace adversarial_2048 {

using CellArray = std::array<std::uint8_t, 16>;

inline constexpr std::size_t kBoardWidth = 4;
inline constexpr std::size_t kCellCount = kBoardWidth * kBoardWidth;
inline constexpr std::uint8_t kMaxExponent = 31;

// Exponents 0-15 live in the original four-bit-per-cell bitboard. One compact
// extension bit per cell represents exponents 16-31. The extension is zero for
// ordinary boards, preserving the lookup-table fast path.
struct Board {
    std::uint64_t packed_exponents{};
    std::uint16_t exponent_high_bits{};

    [[nodiscard]] friend constexpr bool operator==(const Board&, const Board&) = default;
};

enum class Direction : std::uint8_t {
    left,
    right,
    up,
    down,
};

inline constexpr std::array<Direction, 4> kDirections{
    Direction::left,
    Direction::right,
    Direction::up,
    Direction::down,
};

struct MoveResult {
    Board board{};
    std::uint64_t score{};
    bool moved{};
};

[[nodiscard]] Board encode(const CellArray& exponents);
[[nodiscard]] CellArray decode(Board board) noexcept;

[[nodiscard]] std::uint8_t cell_at(Board board, std::size_t index);
[[nodiscard]] Board with_cell(Board board, std::size_t index, std::uint8_t exponent);

[[nodiscard]] Board transpose(Board board) noexcept;
[[nodiscard]] MoveResult move(Board board, Direction direction);

[[nodiscard]] std::size_t empty_count(Board board) noexcept;
[[nodiscard]] std::uint8_t max_exponent(Board board) noexcept;
[[nodiscard]] bool can_move(Board board, Direction direction);
[[nodiscard]] std::array<bool, 4> legal_moves(Board board);
[[nodiscard]] bool is_game_over(Board board);

}  // namespace adversarial_2048
