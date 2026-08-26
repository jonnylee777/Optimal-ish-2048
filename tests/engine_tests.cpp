#include "core/board.hpp"
#include "core/move_tables.hpp"
#include "core/spawn.hpp"

#include <array>
#include <cstdint>
#include <exception>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace a2048 = adversarial_2048;

namespace {

class TestFailure : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

#define CHECK(condition)                                                                          \
    do {                                                                                          \
        if (!(condition)) {                                                                       \
            throw TestFailure(std::string("check failed: ") + #condition + " at line " +         \
                              std::to_string(__LINE__));                                           \
        }                                                                                         \
    } while (false)

using Cells = a2048::CellArray;

[[nodiscard]] std::uint16_t encode_row(const std::array<std::uint8_t, 4>& cells) {
    std::uint16_t row = 0;
    for (std::size_t index = 0; index < cells.size(); ++index) {
        row |= static_cast<std::uint16_t>(cells[index]) << (index * 4U);
    }
    return row;
}

[[nodiscard]] std::uint16_t reverse_row(std::uint16_t row) {
    return static_cast<std::uint16_t>(((row & 0x000FU) << 12U) |
                                      ((row & 0x00F0U) << 4U) |
                                      ((row & 0x0F00U) >> 4U) |
                                      ((row & 0xF000U) >> 12U));
}

[[nodiscard]] a2048::RowMove reference_left(std::uint16_t row) {
    std::array<std::uint8_t, 4> packed{};
    std::size_t packed_size = 0;
    for (std::size_t index = 0; index < packed.size(); ++index) {
        const auto exponent = static_cast<std::uint8_t>((row >> (index * 4U)) & 0xFU);
        if (exponent != 0) {
            packed[packed_size++] = exponent;
        }
    }

    a2048::RowMove result{};
    std::size_t output_index = 0;
    for (std::size_t index = 0; index < packed_size; ++index) {
        auto exponent = packed[index];
        if (index + 1 < packed_size && packed[index + 1] == exponent) {
            if (exponent == 15) {
                result.exponent_overflow = true;
            } else {
                ++exponent;
                result.score += 1U << exponent;
            }
            ++index;
        }
        result.row |= static_cast<std::uint16_t>(exponent) << (output_index * 4U);
        ++output_index;
    }
    return result;
}

[[nodiscard]] a2048::MoveResult reference_move(
    const Cells& input,
    a2048::Direction direction) {
    Cells output{};
    std::uint64_t score = 0;
    const bool horizontal = direction == a2048::Direction::left ||
                            direction == a2048::Direction::right;
    const bool forward = direction == a2048::Direction::left ||
                         direction == a2048::Direction::up;

    for (std::size_t line_index = 0; line_index < a2048::kBoardWidth; ++line_index) {
        std::array<std::uint8_t, 4> compact{};
        std::size_t compact_size = 0;
        for (std::size_t offset = 0; offset < a2048::kBoardWidth; ++offset) {
            const auto position = forward ? offset : a2048::kBoardWidth - 1U - offset;
            const auto cell_index = horizontal
                ? line_index * a2048::kBoardWidth + position
                : position * a2048::kBoardWidth + line_index;
            if (input[cell_index] != 0) {
                compact[compact_size++] = input[cell_index];
            }
        }

        std::size_t output_offset = 0;
        for (std::size_t index = 0; index < compact_size;) {
            auto exponent = compact[index];
            if (index + 1 < compact_size && compact[index + 1] == exponent) {
                ++exponent;
                score += std::uint64_t{1} << exponent;
                index += 2;
            } else {
                ++index;
            }

            const auto position = forward
                ? output_offset
                : a2048::kBoardWidth - 1U - output_offset;
            const auto cell_index = horizontal
                ? line_index * a2048::kBoardWidth + position
                : position * a2048::kBoardWidth + line_index;
            output[cell_index] = exponent;
            ++output_offset;
        }
    }

    const auto input_board = a2048::encode(input);
    const auto output_board = a2048::encode(output);
    return {output_board, score, output_board != input_board};
}

[[nodiscard]] std::uint64_t next_test_random(std::uint64_t& state) {
    state ^= state << 13U;
    state ^= state >> 7U;
    state ^= state << 17U;
    return state;
}

void test_encode_decode() {
    const Cells cells{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 16, 17, 30, 31};
    const auto board = a2048::encode(cells);
    CHECK(a2048::decode(board) == cells);
    CHECK(a2048::cell_at(board, 7) == 7);

    const auto changed = a2048::with_cell(board, 7, 2);
    CHECK(a2048::cell_at(changed, 7) == 2);
    CHECK(a2048::cell_at(changed, 8) == 8);
}

void test_transpose() {
    const Cells cells{16, 2, 3, 4,
                      5, 6, 7, 8,
                      9, 10, 11, 12,
                      13, 14, 31, 0};
    const Cells expected{16, 5, 9, 13,
                         2, 6, 10, 14,
                         3, 7, 11, 31,
                         4, 8, 12, 0};
    const auto board = a2048::encode(cells);
    CHECK(a2048::decode(a2048::transpose(board)) == expected);
    CHECK(a2048::transpose(a2048::transpose(board)) == board);
}

void test_large_tile_merges() {
    const auto pair_32768 = a2048::encode(Cells{15, 15});
    const auto merged_65536 = a2048::move(pair_32768, a2048::Direction::left);
    CHECK(merged_65536.board == a2048::encode(Cells{16}));
    CHECK(merged_65536.score == 65'536);
    CHECK(merged_65536.moved);
    CHECK(a2048::empty_count(merged_65536.board) == 15);

    const auto pair_65536 = a2048::encode(Cells{16, 16});
    const auto merged_131072 = a2048::move(pair_65536, a2048::Direction::right);
    CHECK(merged_131072.board == a2048::encode(Cells{0, 0, 0, 17}));
    CHECK(merged_131072.score == 131'072);

    const auto vertical_pair = a2048::encode(Cells{
        20, 0, 0, 0,
        20, 0, 0, 0,
    });
    const auto vertical_merge = a2048::move(vertical_pair, a2048::Direction::down);
    CHECK(vertical_merge.board == a2048::encode(Cells{
        0, 0, 0, 0,
        0, 0, 0, 0,
        0, 0, 0, 0,
        21, 0, 0, 0,
    }));
    CHECK(vertical_merge.score == (std::uint64_t{1} << 21U));

    bool overflow_detected = false;
    try {
        static_cast<void>(a2048::move(
            a2048::encode(Cells{31, 31}), a2048::Direction::left));
    } catch (const std::overflow_error&) {
        overflow_detected = true;
    }
    CHECK(overflow_detected);
}

void test_row_tables_and_merge_rules() {
    const auto& tables = a2048::move_tables();

    auto result = tables.left(encode_row({1, 0, 1, 1}));
    CHECK(result.row == encode_row({2, 1, 0, 0}));
    CHECK(result.score == 4);

    result = tables.right(encode_row({1, 0, 1, 1}));
    CHECK(result.row == encode_row({0, 0, 1, 2}));
    CHECK(result.score == 4);

    result = tables.left(encode_row({1, 1, 1, 1}));
    CHECK(result.row == encode_row({2, 2, 0, 0}));
    CHECK(result.score == 8);

    result = tables.left(encode_row({2, 2, 3, 0}));
    CHECK(result.row == encode_row({3, 3, 0, 0}));
    CHECK(result.score == 8);

    result = tables.left(encode_row({15, 15, 0, 0}));
    CHECK(result.exponent_overflow);

    for (std::size_t value = 0; value < a2048::MoveTables::kRowCount; ++value) {
        const auto row = static_cast<std::uint16_t>(value);
        const auto expected_left = reference_left(row);
        const auto actual_left = tables.left(row);
        CHECK(actual_left.row == expected_left.row);
        CHECK(actual_left.score == expected_left.score);
        CHECK(actual_left.exponent_overflow == expected_left.exponent_overflow);

        const auto reversed_expected = reference_left(reverse_row(row));
        const auto actual_right = tables.right(row);
        CHECK(actual_right.row == reverse_row(reversed_expected.row));
        CHECK(actual_right.score == reversed_expected.score);
        CHECK(actual_right.exponent_overflow == reversed_expected.exponent_overflow);
    }
}

void test_directional_moves_and_scoring() {
    const auto board = a2048::encode(Cells{
        1, 1, 0, 0,
        1, 0, 1, 0,
        0, 0, 1, 1,
        2, 2, 2, 2,
    });

    const auto left = a2048::move(board, a2048::Direction::left);
    CHECK(left.board == a2048::encode(Cells{
        2, 0, 0, 0,
        2, 0, 0, 0,
        2, 0, 0, 0,
        3, 3, 0, 0,
    }));
    CHECK(left.score == 28);
    CHECK(left.moved);

    const auto right = a2048::move(board, a2048::Direction::right);
    CHECK(right.board == a2048::encode(Cells{
        0, 0, 0, 2,
        0, 0, 0, 2,
        0, 0, 0, 2,
        0, 0, 3, 3,
    }));
    CHECK(right.score == 28);

    const auto up = a2048::move(board, a2048::Direction::up);
    CHECK(up.board == a2048::encode(Cells{
        2, 1, 2, 1,
        2, 2, 2, 2,
        0, 0, 0, 0,
        0, 0, 0, 0,
    }));
    CHECK(up.score == 8);

    const auto down = a2048::move(board, a2048::Direction::down);
    CHECK(down.board == a2048::encode(Cells{
        0, 0, 0, 0,
        0, 0, 0, 0,
        2, 1, 2, 1,
        2, 2, 2, 2,
    }));
    CHECK(down.score == 8);
}

void test_moves_against_reference_implementation() {
    std::uint64_t random_state = 0xA2048B17U;
    constexpr std::size_t board_count = 25'000;

    for (std::size_t board_index = 0; board_index < board_count; ++board_index) {
        Cells cells{};
        for (auto& exponent : cells) {
            const auto sample = next_test_random(random_state);
            exponent = sample % 4U == 0U
                ? 0U
                : static_cast<std::uint8_t>(1U + sample % 20U);
        }

        const auto board = a2048::encode(cells);
        for (const auto direction : a2048::kDirections) {
            const auto expected = reference_move(cells, direction);
            const auto actual = a2048::move(board, direction);
            CHECK(actual.board == expected.board);
            CHECK(actual.score == expected.score);
            CHECK(actual.moved == expected.moved);
        }
    }
}

void test_legal_moves_and_game_over() {
    const auto corner_tile = a2048::encode(Cells{1});
    const auto legal = a2048::legal_moves(corner_tile);
    CHECK(!legal[0]);
    CHECK(legal[1]);
    CHECK(!legal[2]);
    CHECK(legal[3]);
    CHECK(a2048::empty_count(corner_tile) == 15);

    const auto blocked = a2048::encode(Cells{
        1, 2, 1, 2,
        2, 1, 2, 1,
        1, 2, 1, 2,
        2, 1, 2, 1,
    });
    CHECK(a2048::is_game_over(blocked));
    CHECK(a2048::empty_count(blocked) == 0);

    const auto merge_available = a2048::with_cell(blocked, 1, 1);
    CHECK(!a2048::is_game_over(merge_available));
}

void test_spawn_is_deterministic_and_fills_only_empty_cells() {
    a2048::Board first{};
    a2048::Board second{};
    a2048::RandomEngine first_rng(20240818);
    a2048::RandomEngine second_rng(20240818);

    for (std::size_t remaining = a2048::kCellCount; remaining > 0; --remaining) {
        const auto first_spawn = a2048::spawn_random(first, first_rng);
        const auto second_spawn = a2048::spawn_random(second, second_rng);
        CHECK(first_spawn.has_value());
        CHECK(second_spawn.has_value());
        CHECK(first_spawn->cell_index == second_spawn->cell_index);
        CHECK(first_spawn->exponent == second_spawn->exponent);
        CHECK(first == second);
        CHECK(a2048::empty_count(first) == remaining - 1);
        CHECK(first_spawn->exponent == 1 || first_spawn->exponent == 2);
    }

    const auto full_board = first;
    CHECK(!a2048::spawn_random(first, first_rng).has_value());
    CHECK(first == full_board);

    const auto extended_full_board = a2048::encode(Cells{
        16, 16, 16, 16,
        16, 16, 16, 16,
        16, 16, 16, 16,
        16, 16, 16, 16,
    });
    auto extended_copy = extended_full_board;
    CHECK(a2048::empty_count(extended_copy) == 0);
    CHECK(!a2048::spawn_random(extended_copy, first_rng).has_value());
    CHECK(extended_copy == extended_full_board);
}

void test_spawn_distribution() {
    a2048::RandomEngine rng(42);
    std::size_t fours = 0;
    constexpr std::size_t sample_count = 100'000;

    for (std::size_t sample = 0; sample < sample_count; ++sample) {
        a2048::Board board{};
        const auto event = a2048::spawn_random(board, rng);
        CHECK(event.has_value());
        fours += event->exponent == 2 ? 1U : 0U;
    }

    // This broad deterministic check catches probability regressions without
    // making the test sensitive to ordinary sampling variation.
    CHECK(fours > 9'500);
    CHECK(fours < 10'500);
}

}  // namespace

int main() {
    const std::vector<std::pair<std::string, void (*)()>> tests{
        {"encode/decode", test_encode_decode},
        {"transpose", test_transpose},
        {"row tables and merge rules", test_row_tables_and_merge_rules},
        {"directional moves and scoring", test_directional_moves_and_scoring},
        {"differential board moves", test_moves_against_reference_implementation},
        {"large-tile merges", test_large_tile_merges},
        {"legal moves and game over", test_legal_moves_and_game_over},
        {"deterministic spawning", test_spawn_is_deterministic_and_fills_only_empty_cells},
        {"spawn probability", test_spawn_distribution},
    };

    std::size_t failures = 0;
    for (const auto& [name, test] : tests) {
        try {
            test();
            std::cout << "[PASS] " << name << '\n';
        } catch (const std::exception& error) {
            ++failures;
            std::cerr << "[FAIL] " << name << ": " << error.what() << '\n';
        }
    }

    std::cout << tests.size() - failures << '/' << tests.size() << " tests passed\n";
    return failures == 0 ? 0 : 1;
}
