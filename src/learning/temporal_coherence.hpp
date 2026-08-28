#pragma once

#include "learning/ntuple_network.hpp"

#include <cstddef>
#include <filesystem>
#include <vector>

namespace adversarial_2048::learning {

// Temporal coherence (TC) learning: a per-weight adaptive step size.
//
// Plain TD uses one global `alpha` for all 33.7M weights, which is a poor fit
// for an n-tuple network. Some weights sit on patterns the agent sees
// constantly and whose value is already well estimated -- further large steps
// only add noise. Others are rarely visited and still badly wrong. A single
// alpha cannot serve both.
//
// TC (Beal & Smith 1999; applied to 2048 by Jaskowski 2016, arXiv:1604.05085)
// gives each weight its own coefficient derived from the CONSISTENCY of the
// errors it has seen. Per weight it accumulates the signed error sum E and
// the absolute error sum A, then scales that weight's step by
//
//     beta = |E| / A          (defined as 1 while A == 0)
//
// The ratio is a direction-agreement measure, not a magnitude measure:
//
//   * errors always pushing the same way  =>  |E| == A  =>  beta = 1, full
//     step. The weight is still converging and should keep moving.
//   * errors cancelling out               =>  |E| << A  =>  beta -> 0, tiny
//     step. The weight is oscillating around its answer; damp it.
//
// So the step size anneals per weight, on evidence, instead of on a global
// schedule someone had to guess. The paper reports this as the single largest
// improvement in its ablation.
//
// Cost is 2 extra float arrays the size of the weight table: 135 MB of
// weights becomes ~405 MB total. That is training-only state -- it is
// deliberately NOT serialised, since a finished network needs only weights.
class TemporalCoherenceLearner {
public:
    explicit TemporalCoherenceLearner(const NTupleNetwork& network);

    // Applies one TD update of `network` for `afterstate` toward `target`.
    // The TD error is computed internally as target - V(afterstate), and the
    // step is divided by the active weight count as plain TD does.
    //
    // Takes the TARGET rather than the error so the active-weight indices are
    // computed once and used for both reading V and writing the update.
    // Training is memory-bound on a 128 MB weight table, and this is the
    // dominant cost.
    //
    // Returns the mean beta actually applied, purely for diagnostics: it
    // starts near 1.0 and falls as the network settles, which makes "is TC
    // doing anything" observable instead of a matter of faith.
    double update(NTupleNetwork& network, Board afterstate, double target, double alpha);

    // Same update, but with caller-supplied scratch so several threads can call
    // it concurrently. The member scratch buffer is the only thing that makes
    // the overload above single-threaded.
    double update(NTupleNetwork& network, Board afterstate, double target, double alpha,
                  std::vector<std::size_t>& scratch);

    // Use relaxed atomic accesses for the E and A accumulators, for the same
    // reason NTupleNetwork::set_concurrent exists: under Hogwild several threads
    // touch the same entries, and a plain concurrent read/write is a data race.
    // Lost accumulator updates only perturb a step size, which TC is designed to
    // be robust to.
    void set_concurrent(bool concurrent) noexcept { concurrent_ = concurrent; }

    // Mean beta over every weight visited at least once. Cheap enough to log
    // once per evaluation interval, not per move.
    [[nodiscard]] double mean_beta() const;

    // Persist / restore the E and A accumulators.
    //
    // Without this, training cannot be extended: resuming from saved weights
    // would restart every beta at 1.0, so a well-converged network would take
    // full-size steps at alpha=1.0 and lose what it had learned. TC state is
    // therefore what makes long training *incremental* rather than
    // all-or-nothing.
    //
    // Written to a sidecar file rather than into the weight file, so a network
    // for PLAY stays exactly its weight size and nothing that only reads
    // weights needs to know this exists.
    void save(const std::filesystem::path& path) const;
    void load(const std::filesystem::path& path);

    [[nodiscard]] std::size_t bytes() const noexcept {
        return (error_sum_.size() + absolute_error_sum_.size()) * sizeof(float);
    }

private:
    std::vector<float> error_sum_;           // E per weight
    std::vector<float> absolute_error_sum_;  // A per weight
    std::vector<std::size_t> scratch_;       // reused index buffer, avoids per-move allocation
    bool concurrent_{false};

    template <bool Concurrent>
    double update_impl(NTupleNetwork& network, Board afterstate, double target, double alpha,
                       std::vector<std::size_t>& scratch);
};

// Optimistic initialisation: set every weight so that an untouched board
// evaluates to roughly `initial_value`.
//
// V(board) is a sum of exactly `active_weight_count()` weights, so each weight
// is set to initial_value / m. Unvisited patterns then look BETTER than
// visited ones, and greedy action selection is pulled toward them -- built-in
// exploration, without an epsilon parameter. This matters here because the
// papers found explicit exploration actively harmful in 2048 (the spawn
// randomness already supplies plenty), so optimism is the mechanism left.
//
// Passing 0 is a no-op, giving the usual zero initialisation.
void apply_optimistic_initialisation(NTupleNetwork& network, double initial_value);

}  // namespace adversarial_2048::learning
