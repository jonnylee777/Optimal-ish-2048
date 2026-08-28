#pragma once

#include "core/board.hpp"
#include "learning/ntuple_network.hpp"
#include "learning/position_store.hpp"

#include <cstdint>
#include <filesystem>
#include <functional>
#include <optional>
#include <vector>

namespace adversarial_2048::learning {

// Afterstate TD(0) learning for the n-tuple network, following
// Szubert & Jaskowski 2014 and Jaskowski 2016 (arXiv:1604.05085).
//
// The central idea is the AFTERSTATE: the board immediately after sliding and
// merging but BEFORE the random tile spawns. It is deterministic given
// (state, action), so a move is evaluated with one network pass instead of
// averaging over the ~30 possible spawn outcomes.
//
// Policy (greedy, no exploration — the authors found exploration actively
// hurts here, since 2048's own randomness supplies enough):
//
//     a = argmax over legal a of [ reward(s, a) + V(afterstate(s, a)) ]
//
// Update, applied one move late (the target needs the NEXT afterstate):
//
//     delta = reward + V(afterstate_next) - V(afterstate_current)
//     every active weight of afterstate_current += (alpha / m) * delta
//
// and at game over, delta = -V(afterstate_last), since no further reward is
// possible. Dividing alpha by m (the active weight count) is what lets alpha
// be O(1); see the collision caveat on `alpha` below.
struct TrainConfig {
    std::uint64_t games{100000};
    // Per-weight step is alpha / active_weight_count. The papers use 1.0, but
    // note that different symmetric orderings can collide on the same LUT
    // entry (common on sparse early-game boards), which makes a single update
    // overshoot — see tests/ntuple_network_tests.cpp. Hence tunable, not
    // hardcoded.
    double alpha{0.1};
    std::uint64_t seed{1};
    // 0 disables periodic evaluation.
    std::uint64_t evaluate_every{0};
    std::uint64_t evaluation_games{100};
    std::uint64_t checkpoint_every{0};

    // Per-weight adaptive step sizes instead of one global alpha. Costs two
    // extra float arrays the size of the weight table (~270 MB for the default
    // network) and is training-only state. See temporal_coherence.hpp.
    bool temporal_coherence{false};

    // Initialise every weight so an untouched board evaluates to about this
    // score, making unvisited patterns attractive to a greedy policy. 0 keeps
    // the usual zero initialisation. Applied before the first game, and only
    // when not resuming (resuming would erase the loaded weights).
    double optimistic_initial_value{0.0};

    // TD(lambda): blend the 1-step bootstrap with the actual multi-step return.
    //
    //   G_t = r_{t+1} + (1-lambda) * V(s'_{t+1}) + lambda * G_{t+1},   G_terminal = 0
    //
    // lambda = 0 recovers plain TD(0) exactly (the current behaviour);
    // lambda = 1 is the full Monte-Carlo return. Published strong 2048 networks
    // use lambda = 0.5, which this project has never tried.
    //
    // Why it is not just backward replay in another guise — a mistake I made
    // when first dismissing it. Backward replay changes the ORDER updates are
    // applied in, so each target still looks exactly one step ahead. TD(lambda)
    // changes the TARGET itself: it averages returns over many future steps, so
    // credit for a good move reaches back across a whole trajectory rather than
    // seeping one state per episode. Different mechanism, different failure
    // modes.
    //
    // Requires the episode to be buffered, so it forces the backward path on.
    double td_lambda{0.0};

    // Apply an episode's updates in reverse (last afterstate first) once the
    // game has ended, instead of one move behind as the game is played.
    //
    // Both orders implement TD(0); they differ in how fast the terminal signal
    // travels. Playing forward, V(s'_t) is updated toward r + V(s'_{t+1})
    // using the successor's value from BEFORE it learned anything this
    // episode, so "the game ends here" crawls backward one state per episode.
    // Going backward, the successor has already been updated, so a single
    // episode carries the terminal signal along the whole trajectory.
    //
    // This targets a measured defect: every network trained here overvalues
    // the last fifth of a game by roughly 4x (see E8 in
    // docs/experiment-log.md). Costs one Board plus one reward per
    // move of buffering, reused across episodes.
    bool backward_updates{false};

    // Select training actions with EXPECTIMAX SEARCH at this depth instead of
    // 1-ply greedy. 1 keeps the papers' behaviour.
    //
    // Rationale: the deployed agent plays at depth 3, but V is fitted to the
    // states 1-ply self-play visits. Those distributions differ sharply — the
    // depth-3 agent reaches 16384 in 94% of games, far more often than 1-ply
    // training does — so V is least trained exactly where the deployed agent
    // spends its time. Training under search removes that mismatch.
    //
    // Expensive: roughly 27x per move at depth 2, ~660x at depth 3.
    //
    // The transposition table is FORCED OFF during training. It keys on board
    // alone, and weights change after every move, so a cached value from
    // earlier in the same game would be stale — a hazard that does not exist
    // when weights are frozen for evaluation.
    std::uint32_t training_search_depth{1};

    // Fit the network directly to values a DEEP SEARCH produced, instead of
    // (or before) TD self-play. Empty disables it.
    //
    // Ordinary training bootstraps V from its own 1-ply estimates. Distillation
    // instead regresses V onto what depth-4 expectimax concluded, which is a
    // different signal rather than another feature. Searching during training
    // cost 34x and gained nothing, so the search is done offline once and the
    // answers replayed here for free.
    std::vector<ValuedPosition> distill_targets;
    std::uint64_t distill_passes{1};

    // Start this fraction of episodes from a collected late-game position
    // instead of an empty board. 0 keeps ordinary self-play.
    //
    // This targets the measured cause of the score ceiling. An autopsy of 40
    // depth-4 games found 38 died having **never assembled a second 16384** —
    // so the deciding skill is holding a 16384 and rebuilding beneath it, and
    // 1-ply self-play almost never reaches that regime (16384 in 54% of games,
    // 32768 in 0%). Seeding changes WHICH states are updated, which is exactly
    // what simply training longer does not do.
    //
    // Keep it below 1.0: episodes from an empty board are what preserve the
    // early- and mid-game skill the agent already has.
    double seed_position_fraction{0.0};
    std::vector<Board> seed_positions;

    // Where to read/write temporal-coherence accumulators, so training can be
    // extended across runs. Empty means do not persist (TC starts fresh).
    std::filesystem::path temporal_coherence_state;
    bool resume_temporal_coherence{false};
};

struct TrainingSample {
    std::uint64_t games_played{};
    double mean_score{};
    std::uint64_t max_tile{};
};

struct TrainResult {
    std::uint64_t games_played{};
    double mean_score{};        // over the whole run
    std::uint64_t best_score{};
    std::uint64_t highest_tile{};
    std::vector<TrainingSample> learning_curve;
    // Mean temporal-coherence beta at the end of training, or 0 when TC is
    // off. Falls from 1.0 toward 0 as weights stop oscillating, so it makes
    // "is TC actually adapting" observable rather than assumed.
    double final_mean_beta{};
};

// One greedy self-play game using the network as-is (no learning, no search).
// This is the "1-ply" evaluation the papers report.
struct PlayResult {
    std::uint64_t score{};
    std::uint64_t moves{};
    std::uint8_t max_tile_exponent{};
};
[[nodiscard]] PlayResult play_greedy_game(
    const NTupleNetwork& network, std::uint64_t seed);

// Chooses the greedy move for `board`, or nullopt when no move is legal.
[[nodiscard]] std::optional<Direction> greedy_move(
    const NTupleNetwork& network, Board board);

// Trains `network` in place. `on_checkpoint` (if provided) is invoked every
// `checkpoint_every` games with the games completed so far, so the caller can
// persist weights — training runs for hours and must survive interruption.
[[nodiscard]] TrainResult train(
    NTupleNetwork& network, const TrainConfig& config,
    const std::function<void(std::uint64_t)>& on_checkpoint = {});

// Exposed for testing: applies exactly one TD update, returning the delta.
// `next_value` is V(next afterstate), or nullopt at game over.
[[nodiscard]] double apply_td_update(
    NTupleNetwork& network, Board afterstate, double reward,
    std::optional<double> next_value, double alpha);

}  // namespace adversarial_2048::learning
