#include "learning/td_trainer.hpp"

#include "core/random.hpp"
#include "core/spawn.hpp"

#include "learning/temporal_coherence.hpp"

#include "evaluation/n1_evaluator.hpp"
#include "search/expectimax.hpp"

#include <algorithm>
#include <cstddef>
#include <limits>
#include <optional>
#include <vector>

namespace adversarial_2048::learning {
namespace {

struct Candidate {
    Direction direction{};
    Board afterstate{};
    std::uint64_t reward{};
    bool legal{};
    // V(afterstate), carried out of the search that already computed it. The
    // caller needs this value as the next TD target, and recomputing it costs
    // another 32 lookups scattered across a 128 MB table — the dominant cost
    // in training. Stored verbatim rather than recovered as
    // (best_score - reward), because that subtraction would not round-trip
    // exactly and would silently perturb every update.
    double afterstate_value{};
};

// Chooses the action by expectimax search rather than 1-ply greedy. Returns the
// resulting afterstate and its value so the caller's bookkeeping is identical
// either way.
[[nodiscard]] Candidate searched_candidate(
    const NTupleNetwork& network, Board board, std::uint32_t depth) {
    const N1Evaluator evaluator(network);
    ExpectimaxOptions options;
    // See TrainConfig::training_search_depth: weights move every step, so a
    // cached value would be stale within the same game.
    options.use_transposition_table = false;
    Expectimax search(evaluator, options);
    const auto result = search.search(board, depth);
    if (!result.direction.has_value()) {
        return Candidate{};
    }
    const auto moved = move(board, *result.direction);
    return Candidate{*result.direction, moved.board, moved.score, true,
                     network.value(moved.board)};
}

// Evaluates all four moves as afterstates (post-merge, pre-spawn).
[[nodiscard]] Candidate best_candidate(const NTupleNetwork& network, Board board) {
    Candidate best;
    double best_score = -std::numeric_limits<double>::infinity();
    for (const auto direction : kDirections) {
        const auto result = move(board, direction);
        if (!result.moved) {
            continue;
        }
        const auto value = network.value(result.board);
        const auto score = static_cast<double>(result.score) + value;
        if (score > best_score) {
            best_score = score;
            best = Candidate{direction, result.board, result.score, true, value};
        }
    }
    return best;
}

}  // namespace

std::optional<Direction> greedy_move(const NTupleNetwork& network, Board board) {
    const auto candidate = best_candidate(network, board);
    if (!candidate.legal) {
        return std::nullopt;
    }
    return candidate.direction;
}

double apply_td_update(
    NTupleNetwork& network, Board afterstate, double reward,
    std::optional<double> next_value, double alpha) {
    // delta = reward + V(next afterstate) - V(current afterstate); at game
    // over there is no successor and no further reward, so the target is 0.
    const auto current = network.value(afterstate);
    const auto target = next_value.has_value() ? reward + *next_value : 0.0;
    const auto delta = target - current;
    const auto per_weight =
        alpha * delta / static_cast<double>(network.active_weight_count());
    network.update(afterstate, per_weight);
    return delta;
}

PlayResult play_greedy_game(const NTupleNetwork& network, std::uint64_t seed) {
    RandomEngine rng(seed);
    Board board{};
    // Two initial tiles, matching Game's own opening. These cannot fail on an
    // empty board, so the result is deliberately discarded.
    static_cast<void>(spawn_random(board, rng));
    static_cast<void>(spawn_random(board, rng));

    PlayResult result;
    std::uint64_t score = 0;
    while (true) {
        const auto candidate = best_candidate(network, board);
        if (!candidate.legal) {
            break;
        }
        score += candidate.reward;
        board = candidate.afterstate;
        ++result.moves;
        if (!spawn_random(board, rng).has_value()) {
            break;
        }
    }
    result.score = score;
    result.max_tile_exponent = max_exponent(board);
    return result;
}

TrainResult train(
    NTupleNetwork& network, const TrainConfig& config,
    const std::function<void(std::uint64_t)>& on_checkpoint) {
    TrainResult result;
    long double score_total = 0.0L;

    // Both are opt-in and independent: TC changes how big each step is,
    // optimistic initialisation changes where the weights start.
    std::optional<TemporalCoherenceLearner> coherence;
    if (config.temporal_coherence) {
        coherence.emplace(network);
        // Restoring the accumulators is what makes a resumed run continue
        // rather than restart: without them every beta would begin at 1.0 and
        // a converged network would take full-size steps.
        if (config.resume_temporal_coherence && !config.temporal_coherence_state.empty()) {
            coherence->load(config.temporal_coherence_state);
        }
    }
    apply_optimistic_initialisation(network, config.optimistic_initial_value);

    // Applies one update through whichever step-size rule is configured. The
    // TD target is identical either way, so a TC run and a plain run differ in
    // exactly one respect.
    const auto update = [&](Board afterstate, double reward,
                            std::optional<double> next_value) {
        if (!coherence.has_value()) {
            static_cast<void>(
                apply_td_update(network, afterstate, reward, next_value, config.alpha));
            return;
        }
        const auto target = next_value.has_value() ? reward + *next_value : 0.0;
        static_cast<void>(coherence->update(network, afterstate, target, config.alpha));
    };

    // Reused across episodes so backward mode does not allocate per game.
    std::vector<Board> episode_afterstates;
    std::vector<double> episode_rewards;

    for (std::uint64_t game = 0; game < config.games; ++game) {
        // Distinct, reproducible stream per game.
        RandomEngine rng(config.seed + game);
        Board board{};
        static_cast<void>(spawn_random(board, rng));
        static_cast<void>(spawn_random(board, rng));

        // The update is applied one move late, because the target for an
        // afterstate needs the NEXT move's reward and the NEXT afterstate:
        //
        //     V(s'_t)  <-  r_{t+1} + V(s'_{t+1})
        //
        // Note the reward index: it is the reward of the move LEAVING s'_t,
        // not the reward that produced s'_t. Using the latter would make V
        // include a reward already banked before reaching s'_t — which is not
        // a function of s'_t at all, and would be double-counted by the
        // action rule `reward + V(afterstate)`.
        std::optional<Board> pending_afterstate;
        std::uint64_t score = 0;
        episode_afterstates.clear();
        episode_rewards.clear();

        while (true) {
            const auto candidate = config.training_search_depth > 1
                ? searched_candidate(network, board, config.training_search_depth)
                : best_candidate(network, board);
            if (!candidate.legal) {
                break;
            }

            if (config.backward_updates) {
                // Record only; the whole episode is replayed in reverse below.
                // reward[t] is the reward of the move LEAVING afterstate[t],
                // so it is appended one step late, exactly as the forward path
                // consumes it one step late.
                if (!episode_afterstates.empty()) {
                    episode_rewards.push_back(static_cast<double>(candidate.reward));
                }
                episode_afterstates.push_back(candidate.afterstate);
            } else if (pending_afterstate.has_value()) {
                // candidate.afterstate_value is exactly what
                // network.value(candidate.afterstate) would return here: no
                // weight update has been applied since best_candidate ran.
                update(*pending_afterstate, static_cast<double>(candidate.reward),
                       candidate.afterstate_value);
            }

            pending_afterstate = candidate.afterstate;
            score += candidate.reward;
            board = candidate.afterstate;

            if (!spawn_random(board, rng).has_value()) {
                break;
            }
        }

        if (config.backward_updates) {
            if (!episode_afterstates.empty()) {
                // The final afterstate earns nothing further.
                episode_rewards.push_back(0.0);
                // Walk backward so each successor's value is already updated
                // by the time its predecessor reads it.
                for (std::size_t step = episode_afterstates.size(); step-- > 0;) {
                    const auto successor =
                        step + 1 < episode_afterstates.size()
                            ? std::optional<double>(network.value(episode_afterstates[step + 1]))
                            : std::nullopt;
                    update(episode_afterstates[step], episode_rewards[step], successor);
                }
            }
        } else if (pending_afterstate.has_value()) {
            // Terminal: no further move, so no further reward and no successor.
            update(*pending_afterstate, 0.0, std::nullopt);
        }

        ++result.games_played;
        score_total += static_cast<long double>(score);
        result.best_score = std::max(result.best_score, score);
        const auto exponent = max_exponent(board);
        const auto tile = exponent == 0 ? std::uint64_t{0} : std::uint64_t{1} << exponent;
        result.highest_tile = std::max(result.highest_tile, tile);

        if (config.evaluate_every != 0 && (game + 1) % config.evaluate_every == 0) {
            long double evaluation_total = 0.0L;
            std::uint64_t best_tile = 0;
            for (std::uint64_t index = 0; index < config.evaluation_games; ++index) {
                // Evaluation seeds are disjoint from training seeds so the
                // curve measures generalisation, not memorisation.
                const auto played = play_greedy_game(
                    network, 0xE0000000ULL + game * config.evaluation_games + index);
                evaluation_total += static_cast<long double>(played.score);
                const auto played_tile = played.max_tile_exponent == 0
                    ? std::uint64_t{0}
                    : std::uint64_t{1} << played.max_tile_exponent;
                best_tile = std::max(best_tile, played_tile);
            }
            result.learning_curve.push_back(TrainingSample{
                game + 1,
                config.evaluation_games == 0
                    ? 0.0
                    : static_cast<double>(evaluation_total /
                                          static_cast<long double>(config.evaluation_games)),
                best_tile,
            });
        }

        if (config.checkpoint_every != 0 && on_checkpoint &&
            (game + 1) % config.checkpoint_every == 0) {
            on_checkpoint(game + 1);
        }
    }

    if (coherence.has_value() && !config.temporal_coherence_state.empty()) {
        coherence->save(config.temporal_coherence_state);
    }
    result.final_mean_beta = coherence.has_value() ? coherence->mean_beta() : 0.0;
    result.mean_score = result.games_played == 0
        ? 0.0
        : static_cast<double>(score_total / static_cast<long double>(result.games_played));
    return result;
}

}  // namespace adversarial_2048::learning
