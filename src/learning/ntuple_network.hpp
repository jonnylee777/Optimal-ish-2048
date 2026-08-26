#pragma once

#include "core/board.hpp"

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

namespace adversarial_2048::learning {

// N-tuple network state-value function, as used by the strongest published
// 2048 controllers (Szubert & Jaskowski 2014; Jaskowski 2016,
// arXiv:1604.05085). Ported in concept from
// github.com/aszczepanski/2048 (runtime) and
// github.com/wjaskowski/mastering-2048 (training).
//
// A "tuple" is a set of board cells plus one lookup table (LUT) of size
// 16^n indexed by the tile exponents in those cells. The network's value for
// a board is simply the sum of one LUT read per tuple per symmetry:
//
//     V(board) = sum over tuples i, over symmetric orderings j:
//                    lut_i[ index(board, cells_ij) ]
//
// Symmetry is exploited by WEIGHT SHARING, not by transforming the board:
// each tuple's 8 dihedral cell-orderings all index the same LUT. That gives
// 8x generalisation for free and is why the board is never rotated here.
//
// Weights are float32 to match the reference implementations exactly (both
// use `float` LUTs); the accumulator is double.
struct TupleSpec {
    // Row-major cell indices (0..15), i.e. cell = 4*row + column, matching
    // the packed-board layout in core/board.hpp.
    std::vector<std::uint8_t> cells;
};

// The 2014 "large" network: two straight 4-tuples plus two 2x3 rectangles.
// 4 LUTs, 8 orderings each => 32 active weights per evaluation,
// 2*16^4 + 2*16^6 = 33,685,504 weights = ~135 MB as float32.
[[nodiscard]] std::vector<TupleSpec> default_tuple_specs();

// Named tuple configurations, so a training run can choose a network shape
// without a recompile and a weight file records which one it used.
//
//   "default"  4 tuples, 33.7M weights, ~128 MB   (the 2014 network)
//   "large"    5 six-tuples, 83.9M weights, ~320 MB
//
// Throws std::invalid_argument on an unknown name, listing the valid ones —
// a typo must not silently fall back to a different network shape.
[[nodiscard]] std::vector<TupleSpec> named_tuple_specs(const std::string& name);

// Every valid name, for CLI help and error messages.
[[nodiscard]] std::vector<std::string> tuple_configuration_names();

// How many independent weight sets a network keeps, and which one a board
// uses. Multi-stage networks (Jaskowski 2016) exist because a single weight
// set has to serve boards whose remaining-score scales differ by an order of
// magnitude: a 4096 tile early in a game means "lots left to earn", the same
// tile on a jammed board late means "about to die". One table must average the
// two, and this project measured the cost of that averaging — the last fifth of
// a game is overvalued roughly 4x (see ULTIMATE_AGENT_PROGRESS.md).
//
// Splitting by max tile exponent gives each phase its own table. The stage
// function must be MONOTONE NON-DECREASING over a game, so play advances
// through stages and never returns; max exponent satisfies that because tiles
// never shrink.
//
//   stage = clamp(max_exponent - 10, 0, stage_count - 1)
//
// so with 4 stages: <=1024, 2048, 4096, 8192-and-up. Costs stage_count times
// the weight memory and nothing at all in lookup time.
[[nodiscard]] std::size_t stage_of(Board board, std::size_t stage_count) noexcept;

// A whole-board feature, added because the n-tuple design structurally cannot
// express one.
//
// Every tuple sees at most 6 of the 16 cells, so "the board is nearly full and
// I am about to die" is not representable by any single feature -- only
// approximable by the sum. Measurement shows the cost: the strongest network
// here overestimates the remaining score in the final fifth of a game by 5.9x
// (predicts ~137,000, earns ~23,000). That miscalibration also makes search
// actively harmful, because expectimax maximises over inflated estimates.
//
// The fix is one extra table indexed by two cheap global quantities:
//
//     index = empty_count (0..15) * 16 + max_exponent (0..15)
//
// 256 weights -- negligible memory next to hundreds of megabytes of tuples --
// and the first feature in this network that looks at the entire board.
inline constexpr std::size_t kGlobalFeatureSize = 256;

[[nodiscard]] std::size_t global_feature_index(Board board) noexcept;

class NTupleNetwork {
public:
    // Expands each spec into its distinct dihedral orderings and allocates
    // zero-initialised LUTs. `stage_count` > 1 allocates that many independent
    // weight sets; see stage_of(). Throws std::invalid_argument on an empty
    // spec list, an empty/oversized tuple, an out-of-range cell index, or a
    // zero stage count.
    explicit NTupleNetwork(std::vector<TupleSpec> specs, std::size_t stage_count = 1,
                           bool global_features = false);

    [[nodiscard]] bool has_global_features() const noexcept { return global_features_; }

    [[nodiscard]] std::size_t stage_count() const noexcept { return stage_count_; }

    [[nodiscard]] double value(Board board) const noexcept;

    // Adds `per_weight_delta` to every weight active for `board`. This is the
    // whole of the TD update's write side; the caller is responsible for
    // having already divided by the active-weight count.
    void update(Board board, double per_weight_delta) noexcept;

    // Number of weights touched by one evaluation (the paper's `m`). The TD
    // update divides the learning rate by this.
    [[nodiscard]] std::size_t active_weight_count() const noexcept {
        return active_weight_count_;
    }

    // Overwrites `out` with the flat weight indices this board touches, in
    // evaluation order. Exists so a training algorithm can hold per-weight
    // state of its own (temporal coherence keeps two extra accumulators per
    // weight) without duplicating the index arithmetic.
    //
    // Indices may REPEAT when two symmetric orderings collide on the same LUT
    // entry, which happens on symmetric boards. Callers doing per-weight
    // bookkeeping must handle a repeated index as two separate updates, since
    // that is exactly what `update()` does.
    void active_indices(Board board, std::vector<std::size_t>& out) const;

    // Direct weight access for initialisation strategies (e.g. optimistic
    // initialisation). Not for the hot path.
    [[nodiscard]] std::vector<float>& weights() noexcept { return weights_; }
    [[nodiscard]] const std::vector<float>& weights() const noexcept { return weights_; }
    [[nodiscard]] std::size_t total_weight_count() const noexcept;
    [[nodiscard]] const std::vector<TupleSpec>& specs() const noexcept { return specs_; }

    // Binary weight file. The header records the tuple definitions, so a
    // weight file can never be silently paired with a different network
    // shape — mismatches throw rather than producing plausible nonsense.
    void save(const std::filesystem::path& path) const;
    void load(const std::filesystem::path& path);

    // Constructs a network from a weight file's own embedded tuple
    // definitions, rather than requiring the caller to already know the shape.
    //
    // `load()` deliberately validates against an existing network and throws on
    // any mismatch, which is right when you know what you expect. But it also
    // means a consumer can only ever open networks of one shape. This factory
    // makes weight files self-describing, so alternate tuple configurations can
    // be trained and then played without the player being recompiled or told
    // which shape to expect. The same header validation still applies.
    [[nodiscard]] static NTupleNetwork load_from(const std::filesystem::path& path);

    // Stable identifier for provenance in experiment metadata: shape plus a
    // hash of every weight.
    [[nodiscard]] std::string fingerprint() const;

private:
    struct Tuple {
        std::size_t lut_offset{};   // into weights_
        std::size_t lut_size{};     // 16^n
        // One entry per distinct symmetric ordering; each is a list of
        // nibble shifts (4 * cell) in most-significant-first order.
        std::vector<std::vector<std::uint8_t>> ordering_shifts;
    };

    [[nodiscard]] std::size_t index_of(
        std::uint64_t packed, const std::vector<std::uint8_t>& shifts) const noexcept;

    // Offset into weights_ for `board`'s stage. Zero for a single-stage
    // network, so the common case costs nothing.
    [[nodiscard]] std::size_t stage_offset(Board board) const noexcept;

    std::vector<TupleSpec> specs_;
    std::vector<Tuple> tuples_;
    std::vector<float> weights_;
    std::size_t active_weight_count_{};
    std::size_t stage_count_{1};
    std::size_t stage_stride_{};  // weights per stage
    bool global_features_{false};
    std::size_t global_offset_{};  // into a stage, where the global table starts
};

}  // namespace adversarial_2048::learning
