#include "core/board.hpp"

#include "core/move_tables.hpp"

#include <array>
#include <bit>
#include <optional>
#include <stdexcept>

namespace adversarial_2048 {
namespace {

[[nodiscard]] std::array<std::uint8_t, 4> merge_line_left(
    const std::array<std::uint8_t, 4>& line,
    std::uint64_t& score) {
    std::array<std::uint8_t, 4> compact{};
    std::size_t compact_size = 0;
    for (const auto exponent : line) {
        if (exponent != 0) {
            compact[compact_size++] = exponent;
        }
    }

    std::array<std::uint8_t, 4> result{};
    std::size_t output_index = 0;
    for (std::size_t index = 0; index < compact_size;) {
        auto exponent = compact[index];
        if (index + 1 < compact_size && compact[index + 1] == exponent) {
            if (exponent == kMaxExponent) {
                throw std::overflow_error("board exponent range exceeded");
            }
            ++exponent;
            score += std::uint64_t{1} << exponent;
            index += 2;
        } else {
            ++index;
        }
        result[output_index++] = exponent;
    }
    return result;
}

[[nodiscard]] MoveResult move_horizontal_extended(Board board, bool move_left) {
    auto input = decode(board);
    CellArray output{};
    std::uint64_t score = 0;

    for (std::size_t row = 0; row < kBoardWidth; ++row) {
        std::array<std::uint8_t, 4> line{};
        for (std::size_t column = 0; column < kBoardWidth; ++column) {
            const auto source_column = move_left ? column : kBoardWidth - 1U - column;
            line[column] = input[row * kBoardWidth + source_column];
        }

        const auto merged = merge_line_left(line, score);
        for (std::size_t column = 0; column < kBoardWidth; ++column) {
            const auto destination_column = move_left ? column : kBoardWidth - 1U - column;
            output[row * kBoardWidth + destination_column] = merged[column];
        }
    }

    const auto moved_board = encode(output);
    return MoveResult{moved_board, score, moved_board != board};
}

[[nodiscard]] MoveResult move_horizontal(Board board, bool move_left) {
    if (board.exponent_high_bits != 0) {
        return move_horizontal_extended(board, move_left);
    }

    Board moved_board{};
    std::uint64_t score = 0;
    const auto& tables = move_tables();

    for (std::size_t row_index = 0; row_index < kBoardWidth; ++row_index) {
        const auto shift = row_index * 16U;
        const auto row = static_cast<std::uint16_t>(
            (board.packed_exponents >> shift) & 0xFFFFU);
        const RowMove& row_move = move_left ? tables.left(row) : tables.right(row);
        if (row_move.exponent_overflow) {
            return move_horizontal_extended(board, move_left);
        }
        moved_board.packed_exponents |= static_cast<std::uint64_t>(row_move.row) << shift;
        score += row_move.score;
    }

    return MoveResult{moved_board, score, moved_board != board};
}

}  // namespace

Board encode(const CellArray& exponents) {
    Board board{};
    for (std::size_t index = 0; index < exponents.size(); ++index) {
        if (exponents[index] > kMaxExponent) {
            throw std::invalid_argument("cell exponent must be at most 31");
        }
        board.packed_exponents |=
            static_cast<std::uint64_t>(exponents[index] & 0xFU) << (index * 4U);
        if ((exponents[index] & 0x10U) != 0) {
            board.exponent_high_bits |= static_cast<std::uint16_t>(1U << index);
        }
    }
    return board;
}

CellArray decode(Board board) noexcept {
    CellArray exponents{};
    for (std::size_t index = 0; index < exponents.size(); ++index) {
        exponents[index] = static_cast<std::uint8_t>(
            ((board.packed_exponents >> (index * 4U)) & 0xFU) |
            (((board.exponent_high_bits >> index) & 1U) << 4U));
    }
    return exponents;
}

std::uint8_t cell_at(Board board, std::size_t index) {
    if (index >= kCellCount) {
        throw std::out_of_range("board cell index must be less than 16");
    }
    return static_cast<std::uint8_t>(
        ((board.packed_exponents >> (index * 4U)) & 0xFU) |
        (((board.exponent_high_bits >> index) & 1U) << 4U));
}

Board with_cell(Board board, std::size_t index, std::uint8_t exponent) {
    if (index >= kCellCount) {
        throw std::out_of_range("board cell index must be less than 16");
    }
    if (exponent > kMaxExponent) {
        throw std::invalid_argument("cell exponent must be at most 31");
    }

    const auto shift = index * 4U;
    const auto mask = std::uint64_t{0xFU} << shift;
    board.packed_exponents = (board.packed_exponents & ~mask) |
                             (static_cast<std::uint64_t>(exponent & 0xFU) << shift);
    const auto high_mask = static_cast<std::uint16_t>(1U << index);
    board.exponent_high_bits = static_cast<std::uint16_t>(
        (board.exponent_high_bits & ~high_mask) |
        ((exponent & 0x10U) != 0 ? high_mask : 0U));
    return board;
}

Board transpose(Board board) noexcept {
    const auto packed = board.packed_exponents;
    const auto diagonal = packed & 0xF0F00F0FF0F00F0FULL;
    const auto upper_four = packed & 0x0000F0F00000F0F0ULL;
    const auto lower_four = packed & 0x0F0F00000F0F0000ULL;
    const auto swapped_four = diagonal | (upper_four << 12U) | (lower_four >> 12U);

    const auto diagonal_eight = swapped_four & 0xFF00FF0000FF00FFULL;
    const auto upper_eight = swapped_four & 0x00FF00FF00000000ULL;
    const auto lower_eight = swapped_four & 0x00000000FF00FF00ULL;
    Board result{
        diagonal_eight | (upper_eight >> 24U) | (lower_eight << 24U),
        0,
    };

    if (board.exponent_high_bits != 0) {
        for (std::size_t row = 0; row < kBoardWidth; ++row) {
            for (std::size_t column = 0; column < kBoardWidth; ++column) {
                const auto source_bit = static_cast<std::uint16_t>(1U <<
                    (row * kBoardWidth + column));
                if ((board.exponent_high_bits & source_bit) != 0) {
                    result.exponent_high_bits |= static_cast<std::uint16_t>(1U <<
                        (column * kBoardWidth + row));
                }
            }
        }
    }
    return result;
}

namespace {

// Single-transpose vertical move: transpose once so columns read as rows,
// then XOR each column's precomputed delta straight back into the original
// packed board — no transpose back. Returns nullopt to defer to the general
// path when the extension plane is in use or a 15+15 merge overflows the
// four-bit tables. See docs/engine-optimization-notes.md.
[[nodiscard]] std::optional<MoveResult> move_vertical_fast(
    Board board, bool upward) noexcept {
    if (board.exponent_high_bits != 0) {
        return std::nullopt;
    }

    const auto& tables = move_tables();
    const auto transposed = transpose(board).packed_exponents;
    auto packed = board.packed_exponents;
    std::uint64_t score = 0;

    for (std::size_t column = 0; column < kBoardWidth; ++column) {
        const auto line = static_cast<std::uint16_t>(
            (transposed >> (column * 16U)) & 0xFFFFU);
        const auto& entry = upward ? tables.up(line) : tables.down(line);
        if (entry.exponent_overflow) {
            return std::nullopt;
        }
        packed ^= entry.delta << (column * 4U);
        score += entry.score;
    }

    const Board result{packed, 0};
    return MoveResult{result, score, result != board};
}

}  // namespace

MoveResult move(Board board, Direction direction) {
    switch (direction) {
        case Direction::left:
            return move_horizontal(board, true);
        case Direction::right:
            return move_horizontal(board, false);
        case Direction::up: {
            if (const auto fast = move_vertical_fast(board, true)) {
                return *fast;
            }
            auto result = move_horizontal(transpose(board), true);
            result.board = transpose(result.board);
            result.moved = result.board != board;
            return result;
        }
        case Direction::down: {
            if (const auto fast = move_vertical_fast(board, false)) {
                return *fast;
            }
            auto result = move_horizontal(transpose(board), false);
            result.board = transpose(result.board);
            result.moved = result.board != board;
            return result;
        }
    }
    throw std::invalid_argument("unknown move direction");
}

std::size_t empty_count(Board board) noexcept {
    if (board.exponent_high_bits == 0) {
        auto occupied = board.packed_exponents;
        occupied |= occupied >> 2U;
        occupied |= occupied >> 1U;
        occupied &= 0x1111111111111111ULL;
        return kCellCount - static_cast<std::size_t>(std::popcount(occupied));
    }

    std::size_t count = 0;
    for (std::size_t index = 0; index < kCellCount; ++index) {
        const auto low_exponent =
            (board.packed_exponents >> (index * 4U)) & 0xFU;
        const auto high_exponent = (board.exponent_high_bits >> index) & 1U;
        count += low_exponent == 0U && high_exponent == 0U ? 1U : 0U;
    }
    return count;
}

std::uint8_t max_exponent(Board board) noexcept {
    std::uint8_t maximum = 0;
    for (const auto exponent : decode(board)) {
        if (exponent > maximum) {
            maximum = exponent;
        }
    }
    return maximum;
}

bool can_move(Board board, Direction direction) {
    return move(board, direction).moved;
}

std::array<bool, 4> legal_moves(Board board) {
    return {
        can_move(board, Direction::left),
        can_move(board, Direction::right),
        can_move(board, Direction::up),
        can_move(board, Direction::down),
    };
}

bool is_game_over(Board board) {
    const auto moves = legal_moves(board);
    return !moves[0] && !moves[1] && !moves[2] && !moves[3];
}

}  // namespace adversarial_2048
