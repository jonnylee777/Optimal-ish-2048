#include "tablebase/formation.hpp"
#include "tablebase/layer_store.hpp"
#include "tablebase/solver.hpp"

#include <cmath>
#include <cstdint>
#include <exception>
#include <filesystem>
#include <unistd.h>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace tb = adversarial_2048::tablebase;

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

// Scratch directory that cleans up after itself even when a check throws.
class TemporaryDirectory {
public:
    explicit TemporaryDirectory(const std::string& tag) {
        path_ = std::filesystem::temp_directory_path() /
                ("a2048_tb_" + tag + "_" + std::to_string(::getpid()));
        std::filesystem::remove_all(path_);
        std::filesystem::create_directories(path_);
    }
    ~TemporaryDirectory() {
        std::error_code error;
        std::filesystem::remove_all(path_, error);
    }
    TemporaryDirectory(const TemporaryDirectory&) = delete;
    TemporaryDirectory& operator=(const TemporaryDirectory&) = delete;

    [[nodiscard]] const std::filesystem::path& path() const { return path_; }

private:
    std::filesystem::path path_;
};

void test_layer_store_round_trips_boards() {
    const TemporaryDirectory directory("boards");
    const tb::LayerStore store(directory.path(), "t");

    std::vector<std::uint64_t> boards{1, 5, 9, 0xFFFFFFFFFFFFFFFFULL};
    store.write_boards(3, boards);
    CHECK(store.read_boards(3) == boards);

    // A layer never written reads back empty rather than throwing.
    CHECK(store.read_boards(99).empty());

    // Empty layers round-trip too (they are common at low tile sums).
    store.write_boards(4, {});
    CHECK(store.read_boards(4).empty());
}

void test_layer_store_round_trips_solved_bit_exactly() {
    const TemporaryDirectory directory("solved");
    const tb::LayerStore store(directory.path(), "t");

    tb::SolvedLayer solved;
    solved.boards = {2, 7, 1000000};
    // Values chosen to have no exact short decimal form, so a lossy round-trip
    // (e.g. quantizing to uint32) would be caught.
    solved.probabilities = {1.0 / 3.0, 0.1 + 0.2, 0.9999999999999999};
    store.write_solved(11, solved);

    const auto loaded = store.read_solved(11);
    CHECK(loaded.boards == solved.boards);
    CHECK(loaded.probabilities.size() == solved.probabilities.size());
    for (std::size_t index = 0; index < solved.probabilities.size(); ++index) {
        // Bit-exact, not approximate.
        CHECK(loaded.probabilities[index] == solved.probabilities[index]);
    }
    CHECK(loaded.lookup(7).has_value());
    CHECK(!loaded.lookup(8).has_value());
}

void test_partial_writes_are_not_visible() {
    // write_all goes through a .tmp then renames, so a reader must never see a
    // partially written layer. Verify no stray .tmp survives a normal write.
    const TemporaryDirectory directory("atomic");
    const tb::LayerStore store(directory.path(), "t");
    store.write_boards(0, {1, 2, 3});
    for (const auto& entry : std::filesystem::directory_iterator(directory.path())) {
        CHECK(entry.path().extension() != ".tmp");
    }
}

// THE STAGE-B GATE.
//
// The disk-backed solver must reproduce the in-memory solver exactly. The
// in-memory solver is itself validated against an independent brute-force
// recursion (tablebase_solver_tests.cpp), so this chains to a real ground truth
// rather than comparing two copies of the same logic.
void expect_disk_matches_memory(const tb::Formation& formation, std::size_t extra_layers,
                               const std::string& tag) {
    tb::SolveOptions memory_options;
    memory_options.extra_layers = extra_layers;
    const auto in_memory = tb::solve_formation(formation, memory_options);
    CHECK(!in_memory.aborted);
    CHECK(in_memory.total_states > 100);

    const TemporaryDirectory directory(tag);
    tb::DiskSolveOptions disk_options;
    disk_options.solve = memory_options;
    disk_options.directory = directory.path();
    disk_options.prefix = formation.name;
    // Force the compaction path to actually run rather than staying dormant.
    disk_options.compaction_threshold = 4096;
    // Keep the .boards files so the layer contents can be compared directly.
    disk_options.discard_boards_after_solving = false;

    const auto on_disk = tb::solve_formation_to_disk(formation, disk_options);
    CHECK(!on_disk.aborted);
    CHECK(on_disk.total_states == in_memory.total_states);
    CHECK(on_disk.peak_layer_states == in_memory.peak_layer_states);

    // Every layer, every board, every probability — bit-for-bit.
    const tb::LayerStore store(directory.path(), formation.name);
    for (std::size_t layer = 0; layer < in_memory.layers.size(); ++layer) {
        const auto loaded = store.read_solved(layer);
        const auto& expected = in_memory.layers[layer];
        CHECK(loaded.boards == expected.boards);
        CHECK(loaded.probabilities.size() == expected.probabilities.size());
        for (std::size_t index = 0; index < expected.probabilities.size(); ++index) {
            CHECK(loaded.probabilities[index] == expected.probabilities[index]);
        }
    }

    // And the headline number agrees.
    const auto seed_probability =
        in_memory.probability(formation.seeds.front(), formation);
    CHECK(seed_probability.has_value());
    CHECK(std::abs(on_disk.probability_from_seed - *seed_probability) < 1e-15);

    std::cout << "        (" << formation.name << ": " << on_disk.total_states
              << " states, peak layer " << on_disk.peak_layer_states << ", peak disk "
              << on_disk.peak_disk_bytes / 1024 << " KB)\n";
}

void test_disk_matches_memory_2x4() {
    expect_disk_matches_memory(tb::variant_2x4(5), 8, "eq2x4");
}

void test_disk_matches_memory_3x3() {
    expect_disk_matches_memory(tb::variant_3x3(5), 6, "eq3x3");
}

void test_disk_matches_memory_3x4() {
    expect_disk_matches_memory(tb::variant_3x4(4), 6, "eq3x4");
}

void test_disk_budget_guard_aborts() {
    const TemporaryDirectory directory("budget");
    tb::DiskSolveOptions options;
    options.solve.extra_layers = 8;
    options.directory = directory.path();
    options.prefix = "big";
    options.max_disk_bytes = 64 * 1024;  // 64 KB, far too small
    const auto result = tb::solve_formation_to_disk(tb::variant_3x4(7), options);
    CHECK(result.aborted);
    CHECK(!result.abort_reason.empty());
}

void test_boards_are_discarded_when_requested() {
    const TemporaryDirectory directory("discard");
    tb::DiskSolveOptions options;
    options.solve.extra_layers = 6;
    options.directory = directory.path();
    options.prefix = "d";
    options.discard_boards_after_solving = true;
    const auto result = tb::solve_formation_to_disk(tb::variant_2x4(4), options);
    CHECK(!result.aborted);

    std::size_t boards_files = 0;
    std::size_t solved_files = 0;
    for (const auto& entry : std::filesystem::directory_iterator(directory.path())) {
        if (entry.path().extension() == ".boards") {
            ++boards_files;
        } else if (entry.path().extension() == ".solved") {
            ++solved_files;
        }
    }
    CHECK(boards_files == 0);
    CHECK(solved_files > 0);
}

}  // namespace

int main() {
    const std::vector<std::pair<std::string, void (*)()>> tests{
        {"layer store round-trips boards", test_layer_store_round_trips_boards},
        {"layer store round-trips solved bit-exactly", test_layer_store_round_trips_solved_bit_exactly},
        {"no partial writes visible", test_partial_writes_are_not_visible},
        {"GATE: disk == memory (2x4)", test_disk_matches_memory_2x4},
        {"GATE: disk == memory (3x3)", test_disk_matches_memory_3x3},
        {"GATE: disk == memory (3x4)", test_disk_matches_memory_3x4},
        {"disk budget guard aborts", test_disk_budget_guard_aborts},
        {"boards discarded after solving", test_boards_are_discarded_when_requested},
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
