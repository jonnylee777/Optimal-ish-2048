#include "search/expectimax.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <chrono>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace adversarial_2048 {
namespace {

struct SearchDeadlineReached {};

[[nodiscard]] std::size_t mix_hash(std::uint64_t value) noexcept {
    value ^= value >> 30U;
    value *= 0xBF58476D1CE4E5B9ULL;
    value ^= value >> 27U;
    value *= 0x94D049BB133111EBULL;
    value ^= value >> 31U;
    return static_cast<std::size_t>(value);
}

[[nodiscard]] Board reflect_horizontal(Board board) noexcept {
    const auto packed = board.packed_exponents;
    const auto high = board.exponent_high_bits;
    return {
        ((packed & 0x000F000F000F000FULL) << 12U) |
            ((packed & 0x00F000F000F000F0ULL) << 4U) |
            ((packed & 0x0F000F000F000F00ULL) >> 4U) |
            ((packed & 0xF000F000F000F000ULL) >> 12U),
        static_cast<std::uint16_t>(
            ((high & 0x1111U) << 3U) | ((high & 0x2222U) << 1U) |
            ((high & 0x4444U) >> 1U) | ((high & 0x8888U) >> 3U)),
    };
}

[[nodiscard]] Board rotate_clockwise(Board board) noexcept {
    return reflect_horizontal(transpose(board));
}

[[nodiscard]] bool board_less(Board left, Board right) noexcept {
    return left.exponent_high_bits < right.exponent_high_bits ||
           (left.exponent_high_bits == right.exponent_high_bits &&
            left.packed_exponents < right.packed_exponents);
}

[[nodiscard]] Board canonical_symmetry(Board board) noexcept {
    auto best = board;
    auto rotated = board;
    for (std::size_t rotation = 0; rotation < 4; ++rotation) {
        if (board_less(rotated, best)) {
            best = rotated;
        }
        const auto reflected = reflect_horizontal(rotated);
        if (board_less(reflected, best)) {
            best = reflected;
        }
        rotated = rotate_clockwise(rotated);
    }
    return best;
}

}  // namespace

std::uint64_t SearchStatistics::total_nodes() const noexcept {
    return player_nodes + chance_nodes + leaf_evaluations;
}

double SearchStatistics::cache_hit_rate() const noexcept {
    return cache_lookups == 0
        ? 0.0
        : static_cast<double>(cache_hits) / static_cast<double>(cache_lookups);
}

double SearchStatistics::nodes_per_second() const noexcept {
    return elapsed_seconds <= 0.0
        ? 0.0
        : static_cast<double>(total_nodes()) / elapsed_seconds;
}

Expectimax::Expectimax(const Evaluator& evaluator, ExpectimaxOptions options)
    : evaluator_(evaluator), options_(options) {
    if (!std::isfinite(options_.minimum_path_probability) ||
        options_.minimum_path_probability < 0.0 ||
        options_.minimum_path_probability >= 1.0) {
        throw std::invalid_argument(
            "minimum path probability must be finite and in [0, 1)");
    }
    if (!std::isfinite(options_.time_limit_seconds) ||
        options_.time_limit_seconds < 0.0) {
        throw std::invalid_argument("search time limit must be finite and nonnegative");
    }
    if (options_.use_transposition_table) {
        if (options_.transposition_table_capacity == 0) {
            throw std::invalid_argument(
                "transposition table capacity must be greater than zero");
        }
        const auto capacity = std::bit_ceil(
            std::max<std::size_t>(options_.transposition_table_capacity, 4U));
        cache_.resize(capacity);
    }
}

std::size_t Expectimax::CacheKeyHash::operator()(const CacheKey& key) const noexcept {
    auto hash = mix_hash(key.board.packed_exponents);
    hash ^= mix_hash(static_cast<std::uint64_t>(key.board.exponent_high_bits) << 32U);
    hash ^= mix_hash((static_cast<std::uint64_t>(key.depth) << 8U) |
                     static_cast<std::uint8_t>(key.node_type));
    hash ^= mix_hash(key.probability_bits);
    return hash;
}

Expectimax::CacheKey Expectimax::cache_key(
    Board board,
    std::uint32_t depth,
    NodeType node_type,
    double path_probability) const noexcept {
    return {
        options_.use_symmetry_reduction ? canonical_symmetry(board) : board,
        depth,
        node_type,
        options_.minimum_path_probability == 0.0
            ? 0U
            : std::bit_cast<std::uint64_t>(path_probability),
    };
}

void Expectimax::begin_search(bool bounded) {
    if (++cache_generation_ == 0) {
        for (auto& entry : cache_) {
            entry.generation = 0;
        }
        cache_generation_ = 1;
    }
    statistics_ = {};
    deadline_node_counter_ = 0;
    search_start_ = std::chrono::steady_clock::now();
    deadline_ = bounded
        ? search_start_ + std::chrono::duration_cast<std::chrono::steady_clock::duration>(
                              std::chrono::duration<double>(options_.time_limit_seconds))
        : std::chrono::steady_clock::time_point::max();
}

void Expectimax::check_deadline() {
    ++deadline_node_counter_;
    if ((deadline_node_counter_ & 1023U) == 0U &&
        std::chrono::steady_clock::now() >= deadline_) {
        throw SearchDeadlineReached{};
    }
}

SearchResult Expectimax::search(Board board, std::uint32_t depth) {
    if (depth == 0) {
        throw std::invalid_argument("Expectimax depth must be at least one player layer");
    }
    begin_search(false);
    auto result = search_current_generation(board, depth);
    statistics_.elapsed_seconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - search_start_).count();
    result.statistics = statistics_;
    result.completed_depth = depth;
    return result;
}

SearchResult Expectimax::search_iterative(Board board, std::uint32_t maximum_depth) {
    if (maximum_depth == 0) {
        throw std::invalid_argument("Expectimax depth must be at least one player layer");
    }
    if (options_.time_limit_seconds == 0.0) {
        return search(board, maximum_depth);
    }

    begin_search(true);
    SearchResult completed{};
    for (const auto direction : kDirections) {
        const auto moved = move(board, direction);
        if (!moved.moved) {
            continue;
        }
        const auto value = static_cast<double>(moved.score) +
                           evaluator_.evaluate_transition(board, moved.board) +
                           evaluator_.evaluate(moved.board);
        if (!completed.direction.has_value() || value > completed.value) {
            completed.direction = direction;
            completed.value = value;
        }
    }
    for (std::uint32_t depth = 1; depth <= maximum_depth; ++depth) {
        try {
            auto iteration = search_current_generation(board, depth);
            iteration.completed_depth = depth;
            completed = iteration;
        } catch (const SearchDeadlineReached&) {
            break;
        }
    }
    statistics_.elapsed_seconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - search_start_).count();
    completed.statistics = statistics_;
    return completed;
}

double Expectimax::afterstate_value(Board afterstate, std::uint32_t depth) {
    if (depth == 0) {
        throw std::invalid_argument("Expectimax depth must be at least one player layer");
    }
    begin_search(false);
    const auto value = chance_value(afterstate, depth, 1.0);
    statistics_.elapsed_seconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - search_start_).count();
    return value;
}

SearchResult Expectimax::search_current_generation(Board board, std::uint32_t depth) {
    ++statistics_.player_nodes;
    check_deadline();

    std::optional<Direction> best_direction;
    double best_value = -std::numeric_limits<double>::infinity();
    std::array<MoveResult, 4> unique_moves{};
    std::size_t unique_move_count = 0;
    for (const auto direction : kDirections) {
        const auto moved = move(board, direction);
        if (!moved.moved) {
            continue;
        }
        const auto duplicate = std::find_if(
            unique_moves.begin(), unique_moves.begin() + unique_move_count,
            [&moved](const MoveResult& previous) {
                return previous.board == moved.board && previous.score == moved.score;
            });
        if (duplicate != unique_moves.begin() + unique_move_count) {
            continue;
        }
        unique_moves[unique_move_count++] = moved;
        const auto value = static_cast<double>(moved.score) +
                           evaluator_.evaluate_transition(board, moved.board) +
                           chance_value(moved.board, depth, 1.0);
        if (!best_direction.has_value() || value > best_value) {
            best_direction = direction;
            best_value = value;
        }
    }

    if (!best_direction.has_value()) {
        // Root is terminal: the game is over here. Same rule as every other
        // terminal node (E21) -- ask the evaluator what a dead position is
        // worth rather than evaluating the board as if play continued.
        best_value = evaluator_.terminal_value();
    }
    return SearchResult{best_direction, best_value, {}, depth};
}

double Expectimax::player_value(
    Board board, std::uint32_t depth, double path_probability) {
    ++statistics_.player_nodes;
    check_deadline();
    const auto key = cache_key(board, depth, NodeType::player, path_probability);
    if (const auto cached = cached_value(key)) {
        return *cached;
    }

    double best = -std::numeric_limits<double>::infinity();
    bool found_move = false;
    std::array<MoveResult, 4> unique_moves{};
    std::size_t unique_move_count = 0;
    for (const auto direction : kDirections) {
        const auto moved = move(board, direction);
        if (!moved.moved) {
            continue;
        }
        const auto duplicate = std::find_if(
            unique_moves.begin(), unique_moves.begin() + unique_move_count,
            [&moved](const MoveResult& previous) {
                return previous.board == moved.board && previous.score == moved.score;
            });
        if (duplicate != unique_moves.begin() + unique_move_count) {
            continue;
        }
        unique_moves[unique_move_count++] = moved;
        found_move = true;
        best = std::max(
            best,
            static_cast<double>(moved.score) +
                evaluator_.evaluate_transition(board, moved.board) +
                chance_value(moved.board, depth, path_probability));
    }

    if (!found_move) {
        // TERMINAL POSITION: no legal move, so the game ends here and exactly
        // zero further points can be scored.
        //
        // For a value function that predicts remaining score, that value is 0
        // -- NOT whatever the evaluator says about the board. Asking the
        // network here was a real bug with a large cost: it overestimates the
        // late game by roughly 6x, so dead boards (full, big tiles) scored
        // enormously, and search was steered *toward* lines that risk death.
        // Depth 1 never saw it (afterstate semantics expand no spawns), which
        // is why only deeper search was affected.
        //
        // The evaluator decides what that is worth: 0 for a score-predicting
        // value function, "worse than anything achievable" for a positional
        // heuristic whose outputs can go negative.
        best = evaluator_.terminal_value();
    }
    store_value(key, best);
    return best;
}

double Expectimax::chance_value(
    Board board, std::uint32_t depth, double path_probability) {
    ++statistics_.chance_nodes;
    check_deadline();
    const auto key = cache_key(board, depth, NodeType::chance, path_probability);
    if (const auto cached = cached_value(key)) {
        return *cached;
    }

    // For an afterstate value function the leaf IS this board: it is exactly
    // the post-move, pre-spawn position the evaluator was trained on. Forcing
    // one more random spawn here would hand it a board of the wrong kind.
    // Depth still counts player decision layers identically either way.
    if (depth == 1 && evaluator_.semantics() == EvaluationSemantics::afterstate) {
        const auto value = leaf_value(board);
        store_value(key, value);
        return value;
    }

    const auto empties = empty_count(board);
    if (empties == 0) {
        const auto value = depth == 1
            ? leaf_value(board)
            : player_value(board, depth - 1U, path_probability);
        store_value(key, value);
        return value;
    }

    const auto cell_probability = 1.0 / static_cast<double>(empties);
    double expected_value = 0.0;
    for (std::size_t index = 0; index < kCellCount; ++index) {
        const auto shift = index * 4U;
        const auto low_exponent = (board.packed_exponents >> shift) & 0xFU;
        const auto high_exponent = (board.exponent_high_bits >> index) & 1U;
        if (low_exponent != 0 || high_exponent != 0) {
            continue;
        }

        auto with_two = board;
        auto with_four = board;
        with_two.packed_exponents |= std::uint64_t{1} << shift;
        with_four.packed_exponents |= std::uint64_t{2} << shift;
        statistics_.spawn_outcomes += 2;
        const auto two_probability = path_probability * cell_probability * 0.9;
        const auto four_probability = path_probability * cell_probability * 0.1;
        // When a branch is cut (depth exhausted, or path probability below the
        // cutoff) we substitute a static evaluation. WHICH board to evaluate
        // depends on the evaluator, for exactly the reason E1 documents: an
        // afterstate value function scores pre-spawn boards, and handing it
        // `with_two` asks it about a kind of position it never trained on.
        //
        // Depth exhaustion already returns earlier for afterstate evaluators,
        // but the PROBABILITY CUTOFF reaches here at any depth. That path is
        // unused in the fixed-depth regime (cutoff 0) and live in the timed
        // regime, so this was a latent defect waiting for the first timed
        // N-series run.
        const auto cutoff_value = [&](Board spawned) {
            return evaluator_.semantics() == EvaluationSemantics::afterstate
                       ? leaf_value(board)      // the afterstate we arrived at
                       : leaf_value(spawned);   // post-spawn, as H0-H5 expect
        };
        const auto two_value = depth == 1 ||
                two_probability < options_.minimum_path_probability
            ? cutoff_value(with_two)
            : player_value(with_two, depth - 1U, two_probability);
        const auto four_value = depth == 1 ||
                four_probability < options_.minimum_path_probability
            ? cutoff_value(with_four)
            : player_value(with_four, depth - 1U, four_probability);
        expected_value += cell_probability * (0.9 * two_value + 0.1 * four_value);
    }

    store_value(key, expected_value);
    return expected_value;
}

double Expectimax::leaf_value(Board board) {
    ++statistics_.leaf_evaluations;
    return evaluator_.evaluate(board);
}

std::optional<double> Expectimax::cached_value(const CacheKey& key) {
    if (!options_.use_transposition_table) {
        return std::nullopt;
    }
    ++statistics_.cache_lookups;
    constexpr std::size_t probe_count = 4;
    const auto mask = cache_.size() - 1U;
    const auto first = CacheKeyHash{}(key) & mask;
    for (std::size_t probe = 0; probe < probe_count; ++probe) {
        const auto& entry = cache_[(first + probe) & mask];
        if (entry.generation == cache_generation_ && entry.key == key) {
            ++statistics_.cache_hits;
            return entry.value;
        }
    }
    return std::nullopt;
}

void Expectimax::store_value(const CacheKey& key, double value) {
    if (!options_.use_transposition_table) {
        return;
    }

    constexpr std::size_t probe_count = 4;
    const auto mask = cache_.size() - 1U;
    const auto first = CacheKeyHash{}(key) & mask;
    auto replacement = first;
    for (std::size_t probe = 0; probe < probe_count; ++probe) {
        const auto index = (first + probe) & mask;
        const auto& entry = cache_[index];
        if (entry.generation != cache_generation_ || entry.key == key) {
            replacement = index;
            break;
        }
        if (entry.key.depth < cache_[replacement].key.depth) {
            replacement = index;
        }
    }
    cache_[replacement] = CacheEntry{key, value, cache_generation_};
}

}  // namespace adversarial_2048
