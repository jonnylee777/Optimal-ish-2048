#include "core/board.hpp"
#include "tablebase/formation.hpp"
#include "tablebase/solver.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <exception>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace a2048 = adversarial_2048;
namespace tb = adversarial_2048::tablebase;

namespace {

using Cells = a2048::CellArray;

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

#define CHECK_NEAR(a, b, eps)                                                                     \
    do {                                                                                          \
        const double lhs_ = (a);                                                                  \
        const double rhs_ = (b);                                                                  \
        if (std::abs(lhs_ - rhs_) > (eps)) {                                                      \
            throw TestFailure(std::string("expected ") + std::to_string(lhs_) + " ~= " +          \
                              std::to_string(rhs_) + " at line " + std::to_string(__LINE__));    \
        }                                                                                          \
    } while (false)

// THE STAGE-A GATE.
//
// The layered solver (forward BFS by tile-sum layer + backward induction over
// canonicalized states) is compared against a completely separate top-down
// memoized recursion over raw, non-canonicalized boards. The two share only the
// formation predicates. Agreement across every state in the table is real
// evidence that both the DP recurrence and the symmetry reduction are correct;
// a solver agreeing with itself would prove nothing.
void test_layered_dp_matches_brute_force_2x4() {
    const auto formation = tb::variant_2x4(4);  // target tile 16
    tb::SolveOptions options;
    options.extra_layers = 8;
    const auto ceiling = tb::layer_ceiling(formation, options);

    const auto solved = tb::solve_formation(formation, options);
    CHECK(!solved.aborted);
    CHECK(solved.total_states > 1000);

    // Compare EVERY state in the table, not a sample.
    std::size_t compared = 0;
    for (std::size_t layer = 0; layer < solved.layers.size(); ++layer) {
        const auto& entry = solved.layers[layer];
        for (std::size_t index = 0; index < entry.boards.size(); ++index) {
            const auto board = entry.boards[index];
            const auto expected = tb::brute_force_probability(
                board, formation, ceiling, options.spawn_four_probability);
            CHECK_NEAR(entry.probabilities[index], expected, 1e-12);
            ++compared;
        }
    }
    CHECK(compared == solved.total_states);
    std::cout << "        (cross-checked " << compared << " states)\n";
}

// Same gate on a formation whose symmetry group is 8-way rather than 4-way, so
// canonicalization is doing substantially more folding.
void test_layered_dp_matches_brute_force_3x3() {
    const auto formation = tb::variant_3x3(4);  // target tile 16
    tb::SolveOptions options;
    options.extra_layers = 6;
    const auto ceiling = tb::layer_ceiling(formation, options);

    const auto solved = tb::solve_formation(formation, options);
    CHECK(!solved.aborted);
    CHECK(solved.total_states > 1000);

    std::size_t compared = 0;
    for (const auto& entry : solved.layers) {
        for (std::size_t index = 0; index < entry.boards.size(); ++index) {
            const auto expected = tb::brute_force_probability(
                entry.boards[index], formation, ceiling, options.spawn_four_probability);
            CHECK_NEAR(entry.probabilities[index], expected, 1e-12);
            ++compared;
        }
    }
    CHECK(compared == solved.total_states);
    std::cout << "        (cross-checked " << compared << " states)\n";
}

// Every board in a symmetry orbit must get the same probability. If the
// canonicalizer and the mover disagreed about the formation's symmetry, this is
// where it would show up.
void test_symmetric_boards_have_equal_probability() {
    const auto formation = tb::variant_2x4(5);
    tb::SolveOptions options;
    options.extra_layers = 6;
    const auto solved = tb::solve_formation(formation, options);
    CHECK(!solved.aborted);

    const std::array<tb::Symmetry, 4> group{
        tb::Symmetry::identity, tb::Symmetry::reverse_lr,
        tb::Symmetry::reverse_ud, tb::Symmetry::rotate_180};

    std::size_t checked = 0;
    for (const auto& entry : solved.layers) {
        for (std::size_t index = 0; index < entry.boards.size(); ++index) {
            const auto board = entry.boards[index];
            const auto reference = entry.probabilities[index];
            for (const auto symmetry : group) {
                const auto mirrored = tb::apply_symmetry(board, symmetry);
                const auto found = solved.probability(mirrored, formation);
                CHECK(found.has_value());
                CHECK_NEAR(*found, reference, 1e-12);
            }
            ++checked;
            if (checked > 20000) {
                return;  // plenty of coverage; keep the suite fast
            }
        }
    }
}

void test_probabilities_are_valid_and_success_states_are_certain() {
    const auto formation = tb::variant_2x4(5);
    tb::SolveOptions options;
    options.extra_layers = 6;
    const auto solved = tb::solve_formation(formation, options);
    CHECK(!solved.aborted);

    for (const auto& entry : solved.layers) {
        for (std::size_t index = 0; index < entry.boards.size(); ++index) {
            const auto probability = entry.probabilities[index];
            CHECK(probability >= 0.0);
            CHECK(probability <= 1.0);
            // Success states are terminal and deliberately not stored, so no
            // stored board should be one.
            CHECK(!tb::is_success(entry.boards[index], formation.target_exponent,
                                  formation.success_shifts));
        }
    }

    // And a success board reports certainty through the public accessor, even
    // though it is absent from every layer.
    Cells winning{};
    winning[4] = formation.target_exponent;  // a playable cell for 2x4
    const auto board = a2048::encode(winning).packed_exponents |
                       formation.seeds.front();
    const auto found = solved.probability(board, formation);
    CHECK(found.has_value());
    CHECK_NEAR(*found, 1.0, 1e-12);
}

// Harder targets must be no easier to reach. This catches an off-by-one in the
// layer ceiling, which would otherwise silently truncate the search.
void test_harder_targets_are_not_easier() {
    double previous = 1.1;
    for (std::uint8_t target = 4; target <= 7; ++target) {
        const auto formation = tb::variant_2x4(target);
        tb::SolveOptions options;
        options.extra_layers = 8;
        const auto solved = tb::solve_formation(formation, options);
        CHECK(!solved.aborted);
        const auto from_empty = solved.probability(formation.seeds.front(), formation);
        CHECK(from_empty.has_value());
        CHECK(*from_empty <= previous + 1e-12);
        previous = *from_empty;
    }
    // The 2x4 board is genuinely constrained, so by tile 128 it is no longer a
    // certainty — if this were still 1.0 the solver would not be modelling loss.
    CHECK(previous < 1.0);
}

void test_states_outside_the_formation_are_absent() {
    const auto formation = tb::variant_2x4(4);
    tb::SolveOptions options;
    options.extra_layers = 6;
    const auto solved = tb::solve_formation(formation, options);
    CHECK(!solved.aborted);

    // A board with a tile where a wall belongs is not part of the table.
    const auto wall = formation.seeds.front();
    // Clear the wall nibble at cell 0 (row 0 is walled for 2x4) and put a 2 there.
    const auto illegal = (wall & ~(std::uint64_t{0xF})) | std::uint64_t{1};
    CHECK(!solved.probability(illegal, formation).has_value());

    // Every stored board keeps its walls intact.
    for (const auto& entry : solved.layers) {
        for (const auto board : entry.boards) {
            for (std::size_t cell = 0; cell < a2048::kCellCount; ++cell) {
                const bool wall_here = ((wall >> (4 * cell)) & 0xF) == 0xF;
                const bool board_wall = ((board >> (4 * cell)) & 0xF) == 0xF;
                CHECK(wall_here == board_wall);
            }
        }
    }
}

void test_layer_indexing_round_trips() {
    const auto formation = tb::variant_2x4(5);
    tb::SolveOptions options;
    options.extra_layers = 6;
    const auto solved = tb::solve_formation(formation, options);
    CHECK(!solved.aborted);

    // A board stored in layer L must have free tile sum exactly 2L, since that
    // is the indexing invariant the whole backward pass depends on.
    for (std::size_t layer = 0; layer < solved.layers.size(); ++layer) {
        for (const auto board : solved.layers[layer].boards) {
            CHECK(tb::free_tile_sum(board) == 2U * layer);
        }
    }
}

void test_abort_guard_triggers_instead_of_exhausting_memory() {
    const auto formation = tb::variant_3x4(9);  // deliberately large
    tb::SolveOptions options;
    options.extra_layers = 8;
    options.max_total_states = 50'000;  // tiny cap
    const auto solved = tb::solve_formation(formation, options);
    CHECK(solved.aborted);
    CHECK(!solved.abort_reason.empty());
}

}  // namespace

int main() {
    const std::vector<std::pair<std::string, void (*)()>> tests{
        {"GATE: layered DP == brute force (2x4)", test_layered_dp_matches_brute_force_2x4},
        {"GATE: layered DP == brute force (3x3, 8-way symmetry)", test_layered_dp_matches_brute_force_3x3},
        {"symmetric boards share probability", test_symmetric_boards_have_equal_probability},
        {"probabilities valid, successes certain", test_probabilities_are_valid_and_success_states_are_certain},
        {"harder targets are not easier", test_harder_targets_are_not_easier},
        {"out-of-formation states absent", test_states_outside_the_formation_are_absent},
        {"layer indexing round-trips", test_layer_indexing_round_trips},
        {"abort guard triggers", test_abort_guard_triggers_instead_of_exhausting_memory},
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
