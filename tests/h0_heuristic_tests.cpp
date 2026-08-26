#include "core/board.hpp"
#include "evaluation/h0_heuristic.hpp"

#include <algorithm>
#include <cstdint>
#include <exception>
#include <iostream>
#include <random>
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

void test_empty_board_features() {
    const auto board = a2048::encode(Cells{});
    const auto features = a2048::extract_h0_features(board);
    CHECK(features.empty_cells == 16.0);
    CHECK(features.edge_max_bonus == 0.0);
}

void test_center_tile_gets_no_edge_bonus() {
    // Index 5 = row 1, column 1 — one of the four center cells, not an edge.
    Cells cells{};
    cells[5] = 6;
    const auto board = a2048::encode(cells);
    const auto features = a2048::extract_h0_features(board);
    CHECK(features.empty_cells == 15.0);
    CHECK(features.edge_max_bonus == 0.0);
}

void test_non_corner_edge_tile_gets_bonus() {
    // Index 1 = row 0, column 1 — an edge cell, not a corner.
    Cells cells{};
    cells[1] = 7;
    const auto board = a2048::encode(cells);
    const auto features = a2048::extract_h0_features(board);
    CHECK(features.edge_max_bonus == 7.0);
}

void test_corner_tile_gets_bonus() {
    // Index 15 = row 3, column 3 — bottom-right corner.
    Cells cells{};
    cells[15] = 9;
    const auto board = a2048::encode(cells);
    const auto features = a2048::extract_h0_features(board);
    CHECK(features.edge_max_bonus == 9.0);
}

void test_weights_are_separate_and_evaluation_is_deterministic() {
    Cells cells{};
    cells[3] = 5;  // row 0, column 3 — an edge cell.
    const auto board = a2048::encode(cells);

    const a2048::H0Heuristic empty_only({1.0, 0.0});
    CHECK(empty_only.evaluate(board) == 15.0);

    const a2048::H0Heuristic custom({2.0, 3.0});
    const auto features = a2048::extract_h0_features(board);
    const auto expected =
        2.0 * features.empty_cells + 3.0 * features.edge_max_bonus;
    CHECK(custom.evaluate(board) == expected);
    CHECK(custom.evaluate(board) == custom.evaluate(board));
    CHECK(custom.weights().edge_max_bonus == 3.0);
}

void test_is_rotation_invariant() {
    const a2048::H0Heuristic heuristic;
    CHECK(heuristic.is_rotation_invariant());
}

// Independent, decode()-based reimplementation of the same feature
// definitions, used only to cross-check the row-table fast path — written
// separately so this doesn't just compare the optimized code against itself.
[[nodiscard]] a2048::H0Features reference_h0_features(a2048::Board board) {
    const auto cells = a2048::decode(board);
    a2048::H0Features features{};
    std::uint8_t maximum = 0;
    for (const auto exponent : cells) {
        if (exponent == 0) {
            features.empty_cells += 1.0;
        } else {
            maximum = std::max(maximum, exponent);
        }
    }
    if (maximum != 0) {
        for (std::size_t index = 0; index < a2048::kCellCount; ++index) {
            const auto row = index / a2048::kBoardWidth;
            const auto column = index % a2048::kBoardWidth;
            const auto is_edge = row == 0 || row == a2048::kBoardWidth - 1 ||
                                 column == 0 || column == a2048::kBoardWidth - 1;
            if (is_edge && cells[index] == maximum) {
                features.edge_max_bonus = static_cast<double>(maximum);
                break;
            }
        }
    }
    return features;
}

void test_row_table_fast_path_matches_reference_on_random_boards() {
    std::mt19937_64 rng(20260824);
    std::uniform_int_distribution<int> exponent(0, 15);
    for (int trial = 0; trial < 20000; ++trial) {
        Cells cells{};
        for (auto& cell : cells) {
            cell = static_cast<std::uint8_t>(exponent(rng));
        }
        const auto board = a2048::encode(cells);
        const auto fast = a2048::extract_h0_features(board);
        const auto reference = reference_h0_features(board);
        CHECK(fast.empty_cells == reference.empty_cells);
        CHECK(fast.edge_max_bonus == reference.edge_max_bonus);
    }
}

}  // namespace

int main() {
    const std::vector<std::pair<std::string, void (*)()>> tests{
        {"empty board features", test_empty_board_features},
        {"center tile gets no edge bonus", test_center_tile_gets_no_edge_bonus},
        {"non-corner edge tile gets bonus", test_non_corner_edge_tile_gets_bonus},
        {"corner tile gets bonus", test_corner_tile_gets_bonus},
        {"weights are separate and deterministic", test_weights_are_separate_and_evaluation_is_deterministic},
        {"is rotation invariant", test_is_rotation_invariant},
        {"row-table fast path matches reference on 20000 random boards", test_row_table_fast_path_matches_reference_on_random_boards},
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
