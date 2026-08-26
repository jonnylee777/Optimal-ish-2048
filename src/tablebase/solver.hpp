#pragma once

#include "tablebase/formation.hpp"

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <optional>
#include <string>
#include <vector>

namespace adversarial_2048::tablebase {

// Layered endgame solver, ported from game-difficulty/2048EndgameTablebase
// (`native_core/src/BookGenerator.cpp` + `BookSolver.cpp`).
//
// States are POST-MOVE, PRE-SPAWN boards, grouped into layers by total free
// tile sum. A spawn adds 2 or 4 and merges preserve the sum, so layer index
// `sum / 2` only ever increases — the state graph is a DAG and can be solved by
// backward induction:
//
//   P(B) = (1/N_empty) * sum over empty cells c of
//            [ 0.9 * max_d P(move(B + 2@c, d))
//            + 0.1 * max_d P(move(B + 4@c, d)) ]
//
// with P = 1 at success states (terminal, never expanded) and P = 0 wherever no
// legal in-formation move exists. Moves that leave the formation are simply not
// available, which is what confines play to the endgame shape.
//
// This implementation keeps every layer in memory. That is deliberate for the
// small variants used to validate correctness; large 4x4 formations need the
// disk-backed 3-layer window instead (see ROADMAP).
struct SolveOptions {
    double spawn_four_probability{0.1};
    // Layers past this are not generated; states there resolve to P = 0. The
    // reference uses `2^target / 2 + extra_steps`; `extra_layers` is that slack.
    std::size_t extra_layers{48};
    // Hard safety cap. Generation aborts rather than exhausting RAM.
    std::size_t max_total_states{40'000'000};
    bool verbose{};
};

// One solved layer: parallel sorted arrays, binary-searchable by board.
struct SolvedLayer {
    std::vector<std::uint64_t> boards;
    std::vector<double> probabilities;

    [[nodiscard]] std::optional<double> lookup(std::uint64_t board) const noexcept;
};

struct SolveResult {
    std::vector<SolvedLayer> layers;  // indexed by layer number
    std::size_t total_states{};
    std::size_t peak_layer_states{};
    bool aborted{};
    std::string abort_reason;

    // Probability for an arbitrary board, canonicalizing and picking the layer.
    [[nodiscard]] std::optional<double> probability(
        std::uint64_t board, const Formation& formation) const noexcept;
};

[[nodiscard]] SolveResult solve_formation(
    const Formation& formation, const SolveOptions& options = {});

// Independent top-down memoized solver over RAW (non-canonicalized) boards,
// used only to cross-check `solve_formation`. Deliberately shares no code with
// the layered path beyond the formation predicates, so agreement between the
// two is real evidence rather than a tautology.
//
// `max_layer` must match the layered run's own ceiling, otherwise the two
// compute different quantities: both treat states beyond it as P = 0.
[[nodiscard]] double brute_force_probability(
    std::uint64_t board, const Formation& formation, std::size_t max_layer,
    double spawn_four_probability = 0.1);

[[nodiscard]] std::size_t layer_ceiling(
    const Formation& formation, const SolveOptions& options) noexcept;

// Disk-backed variant of `solve_formation`, for tables too large to hold at
// once. Peak memory is bounded by roughly three consecutive layers rather than
// the whole table, which matters because the peak layer is far smaller than the
// total (e.g. 3x4 at target 64: 701K peak vs 5.97M total).
//
// Produces bit-identical probabilities to `solve_formation` — that equivalence
// is asserted in tests/tablebase_disk_tests.cpp, using the in-memory solver
// (itself validated against brute force) as the reference.
struct DiskSolveOptions {
    SolveOptions solve;
    std::filesystem::path directory;
    std::string prefix{"table"};
    // Compact an accumulating buffer once it exceeds this many raw entries.
    // Expansion produces up to ~96 successors per board, most of them
    // duplicates, so periodic compaction is what actually bounds memory.
    std::size_t compaction_threshold{8'000'000};
    // Abort rather than filling the disk.
    std::uintmax_t max_disk_bytes{20ULL * 1024 * 1024 * 1024};
    // Delete each layer's .boards file once it has been solved.
    bool discard_boards_after_solving{true};
};

struct DiskSolveResult {
    std::size_t layer_count{};
    std::size_t total_states{};
    std::size_t peak_layer_states{};
    std::uintmax_t peak_disk_bytes{};
    double probability_from_seed{};
    bool aborted{};
    std::string abort_reason;
};

[[nodiscard]] DiskSolveResult solve_formation_to_disk(
    const Formation& formation, const DiskSolveOptions& options);

}  // namespace adversarial_2048::tablebase
