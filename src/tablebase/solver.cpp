#include "tablebase/solver.hpp"

#include "core/board.hpp"
#include "tablebase/layer_store.hpp"
#include "tablebase/variant_mover.hpp"

#include <algorithm>
#include <array>
#include <iostream>
#include <cstring>
#include <unordered_map>

namespace adversarial_2048::tablebase {
namespace {

// One move under whichever mover the formation needs.
struct MoveOutcome {
    std::uint64_t packed{};
    bool moved{};
};

[[nodiscard]] MoveOutcome apply_move(
    std::uint64_t packed, Direction direction, bool is_variant) noexcept {
    if (is_variant) {
        const auto result = move_variant(packed, direction);
        return MoveOutcome{result.packed, result.moved};
    }
    const auto result = move(Board{packed, 0}, direction);
    return MoveOutcome{result.board.packed_exponents, result.moved};
}

[[nodiscard]] std::size_t layer_of(std::uint64_t packed) noexcept {
    return free_tile_sum(packed) / 2U;
}

void sort_unique(std::vector<std::uint64_t>& boards) {
    std::sort(boards.begin(), boards.end());
    boards.erase(std::unique(boards.begin(), boards.end()), boards.end());
}

}  // namespace

std::optional<double> SolvedLayer::lookup(std::uint64_t board) const noexcept {
    const auto it = std::lower_bound(boards.begin(), boards.end(), board);
    if (it == boards.end() || *it != board) {
        return std::nullopt;
    }
    return probabilities[static_cast<std::size_t>(it - boards.begin())];
}

std::optional<double> SolveResult::probability(
    std::uint64_t board, const Formation& formation) const noexcept {
    if (!matches_formation(board, formation.masks)) {
        return std::nullopt;
    }
    // Success is terminal and certain, so it is never stored in a layer — the
    // value is known without a lookup, at any tile sum.
    if (is_success(board, formation.target_exponent, formation.success_shifts)) {
        return 1.0;
    }
    const auto layer = layer_of(board);
    if (layer >= layers.size()) {
        return std::nullopt;
    }
    return layers[layer].lookup(canonicalize(board, formation.symmetry));
}

std::size_t layer_ceiling(
    const Formation& formation, const SolveOptions& options) noexcept {
    // Reaching a tile of value 2^target needs at least that much tile sum, i.e.
    // layer 2^target / 2. `extra_layers` is the slack the reference calls
    // `extra_steps` (it takes surplus small tiles to assemble the target).
    const auto target_tile = std::size_t{1} << formation.target_exponent;
    return target_tile / 2U + options.extra_layers;
}

SolveResult solve_formation(const Formation& formation, const SolveOptions& options) {
    SolveResult result;
    const auto ceiling = layer_ceiling(formation, options);
    // Exactly the layers we will actually solve. Storing anything past the
    // horizon would leave an unsorted layer behind, and binary-searching that
    // is undefined behaviour.
    const auto layer_count = ceiling + 1;

    std::vector<std::vector<std::uint64_t>> generated(layer_count);
    for (const auto seed : formation.seeds) {
        if (!matches_formation(seed, formation.masks)) {
            continue;
        }
        const auto layer = layer_of(seed);
        if (layer < layer_count) {
            generated[layer].push_back(canonicalize(seed, formation.symmetry));
        }
    }

    // --- forward generation, layer by layer ---
    std::size_t total = 0;
    for (std::size_t layer = 0; layer <= ceiling; ++layer) {
        sort_unique(generated[layer]);
        total += generated[layer].size();
        result.peak_layer_states = std::max(result.peak_layer_states, generated[layer].size());
        if (total > options.max_total_states) {
            result.aborted = true;
            result.abort_reason = "exceeded max_total_states at layer " +
                                  std::to_string(layer) + " (" + std::to_string(total) +
                                  " states)";
            return result;
        }
        if (options.verbose && !generated[layer].empty()) {
            std::cout << "  gen layer " << layer << ": " << generated[layer].size()
                      << " states (total " << total << ")\n";
        }

        for (const auto board : generated[layer]) {
            // Success states are terminal: never expanded, so play stops there.
            if (is_success(board, formation.target_exponent, formation.success_shifts)) {
                continue;
            }
            for (std::size_t cell = 0; cell < kCellCount; ++cell) {
                if (((board >> (4U * cell)) & 0xFULL) != 0) {
                    continue;
                }
                for (const std::uint64_t spawn_exponent : {1ULL, 2ULL}) {
                    const auto spawned = board | (spawn_exponent << (4U * cell));
                    const auto next_layer = layer + static_cast<std::size_t>(spawn_exponent);
                    if (next_layer >= layer_count) {
                        continue;
                    }
                    for (const auto direction : kDirections) {
                        const auto outcome = apply_move(spawned, direction, formation.is_variant);
                        if (!outcome.moved ||
                            !matches_formation(outcome.packed, formation.masks)) {
                            continue;
                        }
                        // Success states are terminal with value 1; storing them
                        // would waste space and make their probability depend on
                        // whether they happen to fall inside the horizon.
                        if (is_success(outcome.packed, formation.target_exponent,
                                       formation.success_shifts)) {
                            continue;
                        }
                        generated[next_layer].push_back(
                            canonicalize(outcome.packed, formation.symmetry));
                    }
                }
            }
        }
    }
    result.total_states = total;

    // --- backward induction ---
    result.layers.resize(layer_count);
    for (std::size_t layer = 0; layer < layer_count; ++layer) {
        result.layers[layer].boards = std::move(generated[layer]);
        result.layers[layer].probabilities.assign(result.layers[layer].boards.size(), 0.0);
    }

    const auto spawn_four = options.spawn_four_probability;
    const auto spawn_two = 1.0 - spawn_four;

    for (std::size_t reverse = 0; reverse <= ceiling; ++reverse) {
        const auto layer = ceiling - reverse;
        auto& current = result.layers[layer];
        for (std::size_t index = 0; index < current.boards.size(); ++index) {
            const auto board = current.boards[index];
            if (is_success(board, formation.target_exponent, formation.success_shifts)) {
                current.probabilities[index] = 1.0;
                continue;
            }

            double accumulated = 0.0;
            std::size_t empty_cells = 0;
            for (std::size_t cell = 0; cell < kCellCount; ++cell) {
                if (((board >> (4U * cell)) & 0xFULL) != 0) {
                    continue;
                }
                ++empty_cells;
                for (const std::uint64_t spawn_exponent : {1ULL, 2ULL}) {
                    const auto spawned = board | (spawn_exponent << (4U * cell));
                    const auto next_layer = layer + static_cast<std::size_t>(spawn_exponent);
                    double best = 0.0;
                    for (const auto direction : kDirections) {
                        const auto outcome =
                            apply_move(spawned, direction, formation.is_variant);
                        if (!outcome.moved ||
                            !matches_formation(outcome.packed, formation.masks)) {
                            continue;
                        }
                        // Terminal win: value 1 regardless of tile sum, so this
                        // is independent of where the horizon happens to fall.
                        if (is_success(outcome.packed, formation.target_exponent,
                                       formation.success_shifts)) {
                            best = 1.0;
                            break;
                        }
                        if (next_layer < result.layers.size()) {
                            const auto found = result.layers[next_layer].lookup(
                                canonicalize(outcome.packed, formation.symmetry));
                            if (found.has_value()) {
                                best = std::max(best, *found);
                            }
                        }
                    }
                    accumulated += best * (spawn_exponent == 1ULL ? spawn_two : spawn_four);
                }
            }
            current.probabilities[index] =
                empty_cells > 0 ? accumulated / static_cast<double>(empty_cells) : 0.0;
        }
        if (options.verbose && !current.boards.empty()) {
            const auto best = *std::max_element(current.probabilities.begin(),
                                                current.probabilities.end());
            std::cout << "  solve layer " << layer << ": " << current.boards.size()
                      << " states, max P = " << best << '\n';
        }
    }
    return result;
}

DiskSolveResult solve_formation_to_disk(
    const Formation& formation, const DiskSolveOptions& options) {
    DiskSolveResult result;
    const auto& solve = options.solve;
    const auto ceiling = layer_ceiling(formation, solve);
    result.layer_count = ceiling + 1;

    const LayerStore store(options.directory, options.prefix);

    // --- forward generation with a rolling 3-layer window ---
    //
    // Layer L receives successors only from L-1 (spawn 2) and L-2 (spawn 4), so
    // once layer L-1 has been expanded, layer L is complete. That is what makes
    // a 3-buffer rolling window sufficient.
    std::vector<std::uint64_t> current;
    std::vector<std::uint64_t> next_one;
    std::vector<std::uint64_t> next_two;

    for (const auto seed : formation.seeds) {
        if (matches_formation(seed, formation.masks) && layer_of(seed) == 0) {
            current.push_back(canonicalize(seed, formation.symmetry));
        }
    }

    const auto compact = [&](std::vector<std::uint64_t>& buffer) {
        if (buffer.size() >= options.compaction_threshold) {
            sort_unique(buffer);
        }
    };

    for (std::size_t layer = 0; layer <= ceiling; ++layer) {
        sort_unique(current);
        store.write_boards(layer, current);
        result.total_states += current.size();
        result.peak_layer_states = std::max(result.peak_layer_states, current.size());

        const auto used = store.total_bytes();
        result.peak_disk_bytes = std::max(result.peak_disk_bytes, used);
        if (used > options.max_disk_bytes) {
            result.aborted = true;
            result.abort_reason = "disk budget exceeded at layer " + std::to_string(layer);
            return result;
        }
        if (solve.verbose && !current.empty()) {
            std::cout << "  gen layer " << layer << ": " << current.size()
                      << " states (total " << result.total_states << ", disk "
                      << used / (1024 * 1024) << " MB)\n";
        }

        for (const auto board : current) {
            if (is_success(board, formation.target_exponent, formation.success_shifts)) {
                continue;
            }
            for (std::size_t cell = 0; cell < kCellCount; ++cell) {
                if (((board >> (4U * cell)) & 0xFULL) != 0) {
                    continue;
                }
                for (const std::uint64_t spawn_exponent : {1ULL, 2ULL}) {
                    const auto spawned = board | (spawn_exponent << (4U * cell));
                    const auto next_layer = layer + static_cast<std::size_t>(spawn_exponent);
                    if (next_layer > ceiling) {
                        continue;
                    }
                    auto& sink = spawn_exponent == 1ULL ? next_one : next_two;
                    for (const auto direction : kDirections) {
                        const auto outcome = apply_move(spawned, direction, formation.is_variant);
                        if (!outcome.moved ||
                            !matches_formation(outcome.packed, formation.masks)) {
                            continue;
                        }
                        if (is_success(outcome.packed, formation.target_exponent,
                                       formation.success_shifts)) {
                            continue;  // terminal, value known without storing
                        }
                        sink.push_back(canonicalize(outcome.packed, formation.symmetry));
                    }
                }
            }
            compact(next_one);
            compact(next_two);
        }

        current = std::move(next_one);
        next_one = std::move(next_two);
        next_two.clear();
        next_two.shrink_to_fit();
    }

    // --- backward solve with a rolling 3-layer window ---
    const auto spawn_four = solve.spawn_four_probability;
    const auto spawn_two = 1.0 - spawn_four;

    SolvedLayer future_one;  // layer + 1
    SolvedLayer future_two;  // layer + 2

    for (std::size_t reverse = 0; reverse <= ceiling; ++reverse) {
        const auto layer = ceiling - reverse;
        SolvedLayer solved;
        solved.boards = store.read_boards(layer);
        solved.probabilities.assign(solved.boards.size(), 0.0);

        for (std::size_t index = 0; index < solved.boards.size(); ++index) {
            const auto board = solved.boards[index];
            double accumulated = 0.0;
            std::size_t empty_cells = 0;
            for (std::size_t cell = 0; cell < kCellCount; ++cell) {
                if (((board >> (4U * cell)) & 0xFULL) != 0) {
                    continue;
                }
                ++empty_cells;
                for (const std::uint64_t spawn_exponent : {1ULL, 2ULL}) {
                    const auto spawned = board | (spawn_exponent << (4U * cell));
                    const auto& future = spawn_exponent == 1ULL ? future_one : future_two;
                    double best = 0.0;
                    for (const auto direction : kDirections) {
                        const auto outcome =
                            apply_move(spawned, direction, formation.is_variant);
                        if (!outcome.moved ||
                            !matches_formation(outcome.packed, formation.masks)) {
                            continue;
                        }
                        if (is_success(outcome.packed, formation.target_exponent,
                                       formation.success_shifts)) {
                            best = 1.0;
                            break;
                        }
                        const auto found =
                            future.lookup(canonicalize(outcome.packed, formation.symmetry));
                        if (found.has_value()) {
                            best = std::max(best, *found);
                        }
                    }
                    accumulated += best * (spawn_exponent == 1ULL ? spawn_two : spawn_four);
                }
            }
            solved.probabilities[index] =
                empty_cells > 0 ? accumulated / static_cast<double>(empty_cells) : 0.0;
        }

        store.write_solved(layer, solved);
        if (options.discard_boards_after_solving) {
            store.remove_boards(layer);
        }
        result.peak_disk_bytes = std::max(result.peak_disk_bytes, store.total_bytes());

        if (solve.verbose && !solved.boards.empty()) {
            const auto best = *std::max_element(solved.probabilities.begin(),
                                                solved.probabilities.end());
            std::cout << "  solve layer " << layer << ": " << solved.boards.size()
                      << " states, max P = " << best << '\n';
        }

        if (layer == 0) {
            for (std::size_t index = 0; index < solved.boards.size(); ++index) {
                result.probability_from_seed =
                    std::max(result.probability_from_seed, solved.probabilities[index]);
            }
        }
        future_two = std::move(future_one);
        future_one = std::move(solved);
    }
    return result;
}

namespace {

double brute_force_recurse(
    std::uint64_t board, const Formation& formation, std::size_t max_layer,
    double spawn_four, std::unordered_map<std::uint64_t, double>& memo) {
    if (is_success(board, formation.target_exponent, formation.success_shifts)) {
        return 1.0;
    }
    // The layered solver computes layers 0..max_layer inclusive and treats
    // anything past that as unreachable. Both must truncate at the SAME point
    // or they are answering different questions.
    if (layer_of(board) > max_layer) {
        return 0.0;
    }
    const auto cached = memo.find(board);
    if (cached != memo.end()) {
        return cached->second;
    }
    // No in-progress placeholder on purpose: every spawn strictly increases the
    // tile sum, so a board can never appear inside its own subtree. If that
    // reasoning were ever wrong we want a stack overflow, not a silent 0.

    const auto spawn_two = 1.0 - spawn_four;
    double accumulated = 0.0;
    std::size_t empty_cells = 0;
    for (std::size_t cell = 0; cell < kCellCount; ++cell) {
        if (((board >> (4U * cell)) & 0xFULL) != 0) {
            continue;
        }
        ++empty_cells;
        for (const std::uint64_t spawn_exponent : {1ULL, 2ULL}) {
            const auto spawned = board | (spawn_exponent << (4U * cell));
            double best = 0.0;
            for (const auto direction : kDirections) {
                const auto outcome = apply_move(spawned, direction, formation.is_variant);
                if (!outcome.moved || !matches_formation(outcome.packed, formation.masks)) {
                    continue;
                }
                best = std::max(best, brute_force_recurse(outcome.packed, formation,
                                                          max_layer, spawn_four, memo));
            }
            accumulated += best * (spawn_exponent == 1ULL ? spawn_two : spawn_four);
        }
    }
    const auto value =
        empty_cells > 0 ? accumulated / static_cast<double>(empty_cells) : 0.0;
    memo[board] = value;
    return value;
}

}  // namespace

double brute_force_probability(
    std::uint64_t board, const Formation& formation, std::size_t max_layer,
    double spawn_four_probability) {
    std::unordered_map<std::uint64_t, double> memo;
    return brute_force_recurse(board, formation, max_layer, spawn_four_probability, memo);
}

}  // namespace adversarial_2048::tablebase
