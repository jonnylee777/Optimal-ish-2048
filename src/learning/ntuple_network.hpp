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
// a game is overvalued roughly 4x (see experiment-log.md).
//
// Splitting by max tile exponent gives each phase its own table. The stage
// function must be MONOTONE NON-DECREASING over a game, so play advances
// through stages and never returns; max exponent satisfies that because tiles
// never shrink.
//
//   stage = clamp(max_exponent - base_exponent + 1, 0, stage_count - 1)
//
// `base_exponent` is the max-tile exponent at which stage 1 begins; everything
// below shares stage 0. Default 10 (1024) preserves the original behaviour.
// Costs stage_count times the weight memory and nothing in lookup time.
//
// The split point matters more than the stage count. An autopsy found 38 of 40
// games die having never assembled a second 16384, so the regime that needs its
// own weights starts at exponent 14 — not at 1024, where the first multi-stage
// attempt split it (and lost 19.4%).
[[nodiscard]] std::size_t stage_of(Board board, std::size_t stage_count,
                                   std::uint8_t base_exponent = 10) noexcept;

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

// How tuple cells are turned into table indices.
//
// `absolute` uses the raw exponent, which is what every published n-tuple
// network does. `relative` shifts every exponent down so the board's largest
// tile maps to 15, clamping at 0.
//
// Why `relative` is worth trying here. Score is set almost entirely by the
// highest tile reached, and an autopsy of 40 games found 38 died having never
// assembled a *second* 16384 — so the deciding skill is rebuilding the snake
// beneath a locked large tile. The agent is already excellent at exactly that
// task one scale lower (it reaches 16384 in 97% of games), but under absolute
// indexing a board of {16384, 8192, 4096} and one of {2048, 1024, 512} occupy
// completely unrelated table entries, so none of that competence transfers.
// Relative indexing makes them the same entry by construction, which reaches
// the rare high-tile regime WITHOUT needing to visit it.
//
// The obvious objection is that it discards absolute scale, and a 32768 board
// genuinely has less game remaining than a same-shaped 2048 board. Pair it with
// `global_features`, whose table is indexed by empty_count x max_exponent and
// therefore carries exactly the discarded information.
//
// MEASURED OUTCOME: rejected. 49,638 against a 135,043 baseline at 100k games,
// never reaching 16384. Making every board look identical regardless of scale
// costs more than the transfer gains -- the network loses track of which regime
// it is in, and one 256-entry table cannot replace what 83.9M weights gave up.
// Kept because it is tested and costs nothing when unused.

// A whole-board STRUCTURAL feature, computed rather than looked up.
//
// The n-tuple network is a lookup table over exact tile values in fixed cell
// groups. It therefore cannot express relational facts — "the large tiles are in
// descending order along an edge", "the biggest tile is cornered" — no matter
// how many weights it has. Those are precisely the concepts that decide 2048,
// and it is why a hand-written evaluator (H5) reaches 109,213 with no learning
// at all while six separate attempts to improve the learned network have all
// tied at ~350,000.
//
// Index packs three cheap quantities:
//
//   descending run along the snake path (0..15)  * 32
//   + largest tile sits in a corner (0/1)        * 16
//   + empty-cell count (0..15)
//
// 512 weights. Deliberately excludes merge-availability, which needs 24
// adjacency checks per call and would slow the hot path measurably.
inline constexpr std::size_t kStructuralFeatureSize = 512;

[[nodiscard]] std::size_t structural_feature_index(Board board) noexcept;

enum class IndexingMode : std::uint8_t {
    absolute,
    relative,
};

class NTupleNetwork {
public:
    // Expands each spec into its distinct dihedral orderings and allocates
    // zero-initialised LUTs. `stage_count` > 1 allocates that many independent
    // weight sets; see stage_of(). Throws std::invalid_argument on an empty
    // spec list, an empty/oversized tuple, an out-of-range cell index, or a
    // zero stage count.
    explicit NTupleNetwork(std::vector<TupleSpec> specs, std::size_t stage_count = 1,
                           bool global_features = false,
                           IndexingMode indexing = IndexingMode::absolute,
                           std::uint8_t stage_base_exponent = 10,
                           bool structural_features = false);

    [[nodiscard]] bool has_structural_features() const noexcept {
        return structural_features_;
    }

    [[nodiscard]] std::uint8_t stage_base_exponent() const noexcept {
        return stage_base_exponent_;
    }

    // Copies this network's stage-0 weights into every other stage.
    //
    // This is "weight promotion": a freshly split network otherwise starts each
    // new stage at zero, so the high-tile stage begins knowing nothing AND
    // receives the least data — which is why the first multi-stage attempt lost
    // 19.4%. Seeding every stage with an already-trained network removes the
    // cold start, leaving each stage to specialise from a competent baseline
    // rather than discover the game from scratch.
    void replicate_stage_zero();

    // Copies another network's TUPLE weights into this one, leaving any extra
    // feature tables (global, structural) at zero.
    //
    // Needed because adding a feature changes the total weight count, so a
    // trained file can no longer be `load()`ed — yet its 83.9M tuple weights are
    // exactly what we want to keep. Without this, testing a new feature would
    // mean retraining from scratch and confounding "does the feature help" with
    // "is 100k games enough".
    //
    // Requires identical tuple specs and stage count; throws otherwise, since a
    // silent partial copy would produce a plausible but meaningless network.
    void adopt_tuple_weights(const NTupleNetwork& source);

    // Size of the tuple-LUT region within one stage, excluding feature tables.
    [[nodiscard]] std::size_t tuple_weight_count() const noexcept;

    [[nodiscard]] IndexingMode indexing() const noexcept { return indexing_; }

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
    // Board -> the packed nibbles the tuples actually index, after high-bit
    // clamping and (optionally) relative normalisation.
    [[nodiscard]] std::uint64_t indexed_packed(Board board) const noexcept;

    std::vector<TupleSpec> specs_;
    std::vector<Tuple> tuples_;
    std::vector<float> weights_;
    std::size_t active_weight_count_{};
    std::size_t stage_count_{1};
    std::size_t stage_stride_{};  // weights per stage
    bool global_features_{false};
    IndexingMode indexing_{IndexingMode::absolute};
    std::uint8_t stage_base_exponent_{10};
    bool structural_features_{false};
    std::size_t structural_offset_{};
    std::size_t global_offset_{};  // into a stage, where the global table starts
};

}  // namespace adversarial_2048::learning
