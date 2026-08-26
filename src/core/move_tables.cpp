#include "core/move_tables.hpp"

#include <array>
#include <cstddef>

namespace adversarial_2048 {
namespace {

constexpr std::uint8_t kMaxExponent = 15;

[[nodiscard]] constexpr std::uint16_t reverse_row(std::uint16_t row) noexcept {
    return static_cast<std::uint16_t>(((row & 0x000FU) << 12U) |
                                      ((row & 0x00F0U) << 4U) |
                                      ((row & 0x0F00U) >> 4U) |
                                      ((row & 0xF000U) >> 12U));
}

[[nodiscard]] RowMove compute_left(std::uint16_t row) noexcept {
    std::array<std::uint8_t, 4> compact{};
    std::size_t compact_size = 0;

    for (std::size_t cell = 0; cell < compact.size(); ++cell) {
        const auto exponent = static_cast<std::uint8_t>((row >> (cell * 4U)) & 0xFU);
        if (exponent != 0) {
            compact[compact_size++] = exponent;
        }
    }

    RowMove result{};
    std::size_t output_cell = 0;
    for (std::size_t index = 0; index < compact_size;) {
        auto exponent = compact[index];
        if (index + 1 < compact_size && compact[index + 1] == exponent) {
            if (exponent == kMaxExponent) {
                result.exponent_overflow = true;
            } else {
                ++exponent;
                result.score += (1U << exponent);
            }
            index += 2;
        } else {
            ++index;
        }

        result.row |= static_cast<std::uint16_t>(
            static_cast<std::uint16_t>(exponent) << (output_cell * 4U));
        ++output_cell;
    }

    return result;
}

// Spreads a 16-bit line's four nibbles down column 0 of a packed board:
// nibble i lands at bit 16*i. Shifting the result left by 4*c moves it to
// column c.
[[nodiscard]] constexpr std::uint64_t unpack_column(std::uint16_t line) noexcept {
    return (static_cast<std::uint64_t>(line & 0x000FU)) |
           (static_cast<std::uint64_t>(line & 0x00F0U) << 12U) |
           (static_cast<std::uint64_t>(line & 0x0F00U) << 24U) |
           (static_cast<std::uint64_t>(line & 0xF000U) << 36U);
}

[[nodiscard]] constexpr ColumnMove to_column_move(
    std::uint16_t line, const RowMove& moved) noexcept {
    return ColumnMove{
        unpack_column(line) ^ unpack_column(moved.row),
        moved.score,
        moved.exponent_overflow,
    };
}

}  // namespace

MoveTables::MoveTables() {
    for (std::size_t value = 0; value < kRowCount; ++value) {
        const auto row = static_cast<std::uint16_t>(value);
        left_[value] = compute_left(row);

        const auto reversed_move = compute_left(reverse_row(row));
        right_[value] = RowMove{
            reverse_row(reversed_move.row),
            reversed_move.score,
            reversed_move.exponent_overflow,
        };

        // A column moving up compacts toward index 0, exactly like a row
        // moving left; moving down mirrors the right-moving row.
        up_[value] = to_column_move(row, left_[value]);
        down_[value] = to_column_move(row, right_[value]);
    }
}

const RowMove& MoveTables::left(std::uint16_t row) const noexcept {
    return left_[row];
}

const RowMove& MoveTables::right(std::uint16_t row) const noexcept {
    return right_[row];
}

const ColumnMove& MoveTables::up(std::uint16_t column) const noexcept {
    return up_[column];
}

const ColumnMove& MoveTables::down(std::uint16_t column) const noexcept {
    return down_[column];
}

const MoveTables& move_tables() {
    static const MoveTables tables;
    return tables;
}

}  // namespace adversarial_2048
