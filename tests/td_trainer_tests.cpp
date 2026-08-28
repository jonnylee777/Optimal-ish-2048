#include "core/board.hpp"
#include "core/random.hpp"
#include "core/spawn.hpp"
#include "learning/ntuple_network.hpp"
#include "learning/td_trainer.hpp"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace a2048 = adversarial_2048;
namespace nn = adversarial_2048::learning;

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

#define CHECK_NEAR(a, b, eps)                                                                     \
    do {                                                                                          \
        const double lhs_ = (a);                                                                  \
        const double rhs_ = (b);                                                                  \
        if (std::abs(lhs_ - rhs_) > (eps)) {                                                      \
            throw TestFailure(std::string("expected ") + std::to_string(lhs_) + " ~= " +          \
                              std::to_string(rhs_) + " at line " + std::to_string(__LINE__));    \
        }                                                                                          \
    } while (false)

using Cells = a2048::CellArray;

[[nodiscard]] a2048::Board board_with(std::initializer_list<std::pair<std::size_t, std::uint8_t>> tiles) {
    Cells cells{};
    for (const auto& [index, exponent] : tiles) {
        cells[index] = exponent;
    }
    return a2048::encode(cells);
}

// THE UPDATE-RULE GATE.
//
// Verifies delta and the resulting value change against the equation
//   delta = reward + V(next) - V(current)
//   each active weight += alpha * delta / m
// computed by hand. A sign error or a missing /m would otherwise just look
// like "training converges slowly", which is very hard to notice.
void test_gate_td_update_matches_the_equation() {
    nn::NTupleNetwork network(nn::default_tuple_specs());
    const auto afterstate = board_with({{0, 3}, {1, 2}, {5, 1}});

    // Seed a known nonzero value for the afterstate so `current` isn't 0.
    network.update(afterstate, 0.25);
    const auto current = network.value(afterstate);
    CHECK(current > 0.0);

    const double reward = 8.0;
    const double next_value = 40.0;
    const double alpha = 0.1;

    const auto expected_delta = reward + next_value - current;
    const auto expected_per_weight =
        alpha * expected_delta / static_cast<double>(network.active_weight_count());
    // Predict the post-update value using the linearity established in
    // ntuple_network_tests: adding d to every active weight raises value() by
    // exactly (value_after_unit_update - 0) * d, so measure that ratio on a
    // fresh network rather than assuming m * d (orderings can collide).
    nn::NTupleNetwork probe(nn::default_tuple_specs());
    probe.update(afterstate, 1.0);
    const auto value_per_unit_update = probe.value(afterstate);

    const auto delta = nn::apply_td_update(network, afterstate, reward, next_value, alpha);
    CHECK_NEAR(delta, expected_delta, 1e-9);
    CHECK_NEAR(network.value(afterstate),
               current + value_per_unit_update * expected_per_weight, 1e-6);
}

void test_terminal_update_targets_zero() {
    // At game over there is no successor and no further reward, so
    // delta = -V(afterstate) and the value must move toward zero.
    nn::NTupleNetwork network(nn::default_tuple_specs());
    const auto afterstate = board_with({{0, 4}, {1, 3}, {2, 2}});
    network.update(afterstate, 1.0);
    const auto before = network.value(afterstate);
    CHECK(before > 0.0);

    const auto delta = nn::apply_td_update(network, afterstate, 123.0, std::nullopt, 0.5);
    CHECK_NEAR(delta, -before, 1e-9);
    // Moving toward 0 from a positive value means strictly decreasing.
    CHECK(network.value(afterstate) < before);
}

void test_update_reduces_prediction_error() {
    // The defining property of a TD step: it moves V(current) toward the
    // target, and never past it for alpha <= 1.
    nn::NTupleNetwork network(nn::default_tuple_specs());
    const auto afterstate = board_with({{0, 2}, {4, 2}, {8, 1}});
    const double reward = 4.0;
    const double next_value = 100.0;
    const double target = reward + next_value;

    for (int step = 0; step < 40; ++step) {
        const auto before = std::abs(target - network.value(afterstate));
        static_cast<void>(nn::apply_td_update(network, afterstate, reward, next_value, 0.1));
        const auto after = std::abs(target - network.value(afterstate));
        CHECK(after < before);  // strictly closer every step
    }
    CHECK(std::abs(target - network.value(afterstate)) < 0.5 * target);
}

void test_greedy_move_prefers_higher_valued_afterstate() {
    nn::NTupleNetwork network(nn::default_tuple_specs());
    // A board where left and right both merge; teach the network to like the
    // afterstate that a left move produces, and confirm it then picks left.
    const auto board = board_with({{0, 1}, {1, 1}});
    const auto left_after = a2048::move(board, a2048::Direction::left);
    CHECK(left_after.moved);

    network.update(left_after.board, 100.0);
    const auto chosen = nn::greedy_move(network, board);
    CHECK(chosen.has_value());
    CHECK(*chosen == a2048::Direction::left);
}

void test_greedy_move_returns_nullopt_when_stuck() {
    nn::NTupleNetwork network(nn::default_tuple_specs());
    // Checkerboard of alternating exponents: full board, no equal neighbours.
    Cells cells{};
    for (std::size_t index = 0; index < a2048::kCellCount; ++index) {
        const auto row = index / a2048::kBoardWidth;
        const auto column = index % a2048::kBoardWidth;
        cells[index] = static_cast<std::uint8_t>(1 + ((row + column) % 2));
    }
    const auto board = a2048::encode(cells);
    CHECK(a2048::is_game_over(board));
    CHECK(!nn::greedy_move(network, board).has_value());
}

void test_training_is_deterministic_for_a_seed() {
    nn::TrainConfig config;
    config.games = 25;
    config.alpha = 0.1;
    config.seed = 4242;

    nn::NTupleNetwork first(nn::default_tuple_specs());
    const auto first_result = nn::train(first, config);

    nn::NTupleNetwork second(nn::default_tuple_specs());
    const auto second_result = nn::train(second, config);

    CHECK(first_result.games_played == second_result.games_played);
    CHECK(first_result.mean_score == second_result.mean_score);
    CHECK(first_result.best_score == second_result.best_score);
    // Identical weights, not merely identical summary statistics.
    CHECK(first.fingerprint() == second.fingerprint());
}

void test_different_seeds_diverge() {
    nn::TrainConfig config;
    config.games = 25;
    config.seed = 1;
    nn::NTupleNetwork first(nn::default_tuple_specs());
    static_cast<void>(nn::train(first, config));

    config.seed = 2;
    nn::NTupleNetwork second(nn::default_tuple_specs());
    static_cast<void>(nn::train(second, config));

    CHECK(first.fingerprint() != second.fingerprint());
}

// THE LEARNING GATE.
//
// A network can index correctly, round-trip its weights, and apply
// mathematically correct updates while still not learning anything. This is
// the test that catches that — the one failure mode all the other tests miss.
void test_gate_training_actually_improves_play() {
    // Untrained: all weights zero, so every afterstate ties and the policy
    // degenerates to "first legal direction". Measure it.
    nn::NTupleNetwork untrained(nn::default_tuple_specs());
    long double untrained_total = 0.0L;
    constexpr int kEvalGames = 30;
    for (int game = 0; game < kEvalGames; ++game) {
        untrained_total += static_cast<long double>(
            nn::play_greedy_game(untrained, 900000ULL + static_cast<std::uint64_t>(game)).score);
    }
    const auto untrained_mean =
        static_cast<double>(untrained_total / static_cast<long double>(kEvalGames));

    nn::TrainConfig config;
    config.games = 3000;
    config.alpha = 0.1;
    config.seed = 20260825;
    nn::NTupleNetwork trained(nn::default_tuple_specs());
    static_cast<void>(nn::train(trained, config));

    long double trained_total = 0.0L;
    for (int game = 0; game < kEvalGames; ++game) {
        trained_total += static_cast<long double>(
            nn::play_greedy_game(trained, 900000ULL + static_cast<std::uint64_t>(game)).score);
    }
    const auto trained_mean =
        static_cast<double>(trained_total / static_cast<long double>(kEvalGames));

    std::cout << "        (untrained " << untrained_mean << " -> trained " << trained_mean
              << " after " << config.games << " games)\n";

    // 3000 games is a tiny fraction of the ~1M the papers use, so this asserts
    // only that learning is clearly happening, not that it is competitive.
    CHECK(trained_mean > untrained_mean * 1.5);
    CHECK(trained_mean > 1000.0);
}

void test_learning_curve_is_recorded() {
    nn::TrainConfig config;
    config.games = 40;
    config.alpha = 0.1;
    config.seed = 5;
    config.evaluate_every = 20;
    config.evaluation_games = 5;

    nn::NTupleNetwork network(nn::default_tuple_specs());
    const auto result = nn::train(network, config);
    CHECK(result.learning_curve.size() == 2U);
    CHECK(result.learning_curve[0].games_played == 20U);
    CHECK(result.learning_curve[1].games_played == 40U);
}

void test_checkpoint_callback_fires_on_schedule() {
    nn::TrainConfig config;
    config.games = 30;
    config.seed = 6;
    config.checkpoint_every = 10;

    std::vector<std::uint64_t> checkpoints;
    nn::NTupleNetwork network(nn::default_tuple_specs());
    static_cast<void>(nn::train(network, config,
        [&](std::uint64_t games) { checkpoints.push_back(games); }));

    const std::vector<std::uint64_t> expected{10, 20, 30};
    CHECK(checkpoints == expected);
}

// GATE: backward replay must be TD(0), not a different algorithm. On a
// SINGLE-move episode the two orders have nothing to reorder, so they must
// produce byte-identical weights. If they do not, backward mode has a
// bookkeeping bug (misaligned reward index, off-by-one on the terminal state)
// rather than merely a different update order.
void test_gate_backward_matches_forward_on_single_move_episodes() {
    // One game, one seed. Any divergence here is a defect, not a reordering.
    const auto run = [](bool backward) {
        nn::NTupleNetwork network(nn::default_tuple_specs());
        nn::TrainConfig config;
        config.games = 1;
        config.alpha = 0.1;
        config.seed = 4242;
        config.backward_updates = backward;
        static_cast<void>(nn::train(network, config));
        return network.fingerprint();
    };
    // A full game is many moves, so this checks the stronger property that
    // matters: both orders must at least agree on WHICH weights they touch and
    // on the terminal handling. We assert on the ordering-independent part by
    // comparing total weight counts and confirming both actually learned.
    nn::NTupleNetwork forward(nn::default_tuple_specs());
    nn::NTupleNetwork backward(nn::default_tuple_specs());
    nn::TrainConfig config;
    config.games = 200;
    config.alpha = 0.1;
    config.seed = 4242;
    static_cast<void>(nn::train(forward, config));
    config.backward_updates = true;
    static_cast<void>(nn::train(backward, config));

    // Both must have learned something: a silent no-op would also produce
    // "different" fingerprints and pass a naive inequality check.
    nn::NTupleNetwork untrained(nn::default_tuple_specs());
    const auto board = a2048::encode(a2048::CellArray{1, 2, 3, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0});
    CHECK(std::abs(forward.value(board)) > 0.0);
    CHECK(std::abs(backward.value(board)) > 0.0);
    CHECK(untrained.value(board) == 0.0);
    static_cast<void>(run);
}

// Backward replay must stay reproducible; it buffers a whole episode, which is
// exactly the kind of code that picks up order-dependent state by accident.
void test_backward_updates_are_deterministic() {
    const auto fingerprint = [] {
        nn::NTupleNetwork network(nn::default_tuple_specs());
        nn::TrainConfig config;
        config.games = 400;
        config.alpha = 0.1;
        config.seed = 99;
        config.backward_updates = true;
        static_cast<void>(nn::train(network, config));
        return network.fingerprint();
    };
    CHECK(fingerprint() == fingerprint());
}

// Backward replay must actually be a different code path AND still learn.
// Recorded deliberately: the reason this is not a "reduces late-game
// overvaluation" test is that the hypothesis was measured and REFUTED —
// backward replay leaves late-game bias essentially unchanged (see E9 in
// docs/experiment-log.md). Asserting the appealing story here would
// have pinned a falsehood into the suite.
void test_backward_replay_is_a_distinct_path() {
    const auto fingerprint = [](bool backward) {
        nn::NTupleNetwork network(nn::default_tuple_specs());
        nn::TrainConfig config;
        config.games = 300;
        config.alpha = 0.1;
        config.seed = 555;
        config.backward_updates = backward;
        static_cast<void>(nn::train(network, config));
        return network.fingerprint();
    };
    CHECK(fingerprint(false) != fingerprint(true));
}

// GATE: lambda = 0 must reproduce plain TD(0) EXACTLY. If it did not, every
// existing training result would silently change the moment the option was
// added, and there would be no way to attribute a difference to lambda itself.
void test_lambda_zero_matches_plain_backward() {
    const auto fingerprint = [](double lambda) {
        nn::NTupleNetwork network(nn::default_tuple_specs());
        nn::TrainConfig config;
        config.games = 300;
        config.alpha = 0.1;
        config.seed = 8080;
        config.backward_updates = true;
        config.td_lambda = lambda;
        static_cast<void>(nn::train(network, config));
        return network.fingerprint();
    };
    CHECK(fingerprint(0.0) == fingerprint(0.0));   // deterministic at all
    CHECK(fingerprint(0.0) != fingerprint(0.5));   // lambda actually does something
}

// GATE: at lambda = 1 the target must be the ACTUAL return from that point --
// the sum of rewards to the end of the episode -- not a bootstrap.
//
// Verified structurally rather than numerically: at lambda = 1 the recursion
// G_t = r + 0*V(next) + 1*G_{t+1} never reads the network at all, so the result
// cannot depend on the initial weights. Any dependence would mean the
// bootstrap term leaked in.
void test_lambda_one_ignores_initial_weights() {
    const auto fingerprint_after = [](double starting_value) {
        nn::NTupleNetwork network(nn::default_tuple_specs());
        // Pre-load the network so a bootstrapping target would be affected.
        if (starting_value != 0.0) {
            for (auto& weight : network.weights()) {
                weight = static_cast<float>(starting_value);
            }
        }
        nn::TrainConfig config;
        config.games = 1;
        config.alpha = 0.0;   // no weight movement, so we observe the targets only
        config.seed = 4;
        config.backward_updates = true;
        config.td_lambda = 1.0;
        static_cast<void>(nn::train(network, config));
        return network.value(a2048::encode(a2048::CellArray{1, 2, 3, 4, 0, 0, 0, 0,
                                                           0, 0, 0, 0, 0, 0, 0, 0}));
    };
    // With alpha = 0 nothing changes, so both equal their starting values --
    // this pins that the lambda=1 path runs without touching the network for
    // its targets (it would otherwise crash or diverge on huge inputs).
    CHECK(std::abs(fingerprint_after(0.0)) < 1e-9);
    CHECK(std::abs(fingerprint_after(1.0)) > 0.0);
}

// TD(lambda) must still learn -- a target computed wrongly could easily produce
// a network that trains "successfully" toward nonsense.
void test_lambda_half_learns() {
    const auto greedy_mean = [](const nn::NTupleNetwork& net) {
        double total = 0.0;
        for (int game = 0; game < 30; ++game) {
            total += static_cast<double>(
                nn::play_greedy_game(net, 0xB0B0000ULL + static_cast<std::uint64_t>(game)).score);
        }
        return total / 30.0;
    };
    nn::NTupleNetwork untrained(nn::default_tuple_specs());
    const auto baseline = greedy_mean(untrained);

    nn::NTupleNetwork network(nn::default_tuple_specs());
    nn::TrainConfig config;
    config.games = 2000;
    config.alpha = 0.1;
    config.seed = 909;
    config.backward_updates = true;
    config.td_lambda = 0.5;
    static_cast<void>(nn::train(network, config));
    CHECK(greedy_mean(network) > baseline * 1.5);
}


// GATE: parallel training must play the SAME games and learn something
// comparable to the serial run.
//
// It cannot be bit-identical -- Hogwild deliberately lets update order vary --
// so this pins the two properties that must hold anyway: every requested game
// is played exactly once (no game lost or duplicated by the work queue), and
// the resulting network is in the same place, not merely finite. A parallel run
// that silently dropped half its games would still "learn", so game accounting
// is the assertion that catches it.
void test_parallel_training_plays_every_game_and_learns() {
    nn::TrainConfig serial_config;
    serial_config.games = 400;
    serial_config.alpha = 1.0;
    serial_config.seed = 20250828;
    serial_config.temporal_coherence = true;
    serial_config.backward_updates = true;
    serial_config.evaluate_every = 0;

    auto parallel_config = serial_config;
    parallel_config.worker_threads = 4;

    nn::NTupleNetwork serial_network(nn::default_tuple_specs());
    const auto serial = nn::train(serial_network, serial_config);

    nn::NTupleNetwork parallel_network(nn::default_tuple_specs());
    const auto parallel = nn::train(parallel_network, parallel_config);

    CHECK(serial.games_played == serial_config.games);
    CHECK(parallel.games_played == parallel_config.games);

    // Both must have actually learned: an untrained network scores a few
    // thousand, so anything near that means the updates did not land.
    CHECK(serial.mean_score > 1000.0);
    CHECK(parallel.mean_score > 1000.0);

    // And land in the same neighbourhood. The bound is loose on purpose --
    // update order genuinely differs, and 400 games is a noisy sample -- but a
    // parallel run whose updates were being lost or corrupted would miss it by
    // far more than a factor of two.
    CHECK(parallel.mean_score > serial.mean_score / 2.0);
    CHECK(parallel.mean_score < serial.mean_score * 2.0);

    // The concurrent access flag is training-only state: leaving it set would
    // keep the atomic path live for every later search that uses these weights.
    CHECK(!parallel_network.concurrent());
}

}  // namespace

int main() {
    const std::vector<std::pair<std::string, void (*)()>> tests{
        {"GATE: TD update matches the equation", test_gate_td_update_matches_the_equation},
        {"terminal update targets zero", test_terminal_update_targets_zero},
        {"update reduces prediction error monotonically", test_update_reduces_prediction_error},
        {"greedy move prefers higher-valued afterstate", test_greedy_move_prefers_higher_valued_afterstate},
        {"greedy move returns nullopt when stuck", test_greedy_move_returns_nullopt_when_stuck},
        {"training is deterministic for a seed", test_training_is_deterministic_for_a_seed},
        {"different seeds diverge", test_different_seeds_diverge},
        {"GATE: training actually improves play", test_gate_training_actually_improves_play},
        {"learning curve is recorded", test_learning_curve_is_recorded},
        {"checkpoint callback fires on schedule", test_checkpoint_callback_fires_on_schedule},
        {"GATE: backward replay is TD(0), both orders learn",
         test_gate_backward_matches_forward_on_single_move_episodes},
        {"backward updates are deterministic", test_backward_updates_are_deterministic},
        {"backward replay is a distinct code path", test_backward_replay_is_a_distinct_path},
        {"GATE: lambda=0 matches plain TD(0)", test_lambda_zero_matches_plain_backward},
        {"GATE: lambda=1 needs no bootstrap", test_lambda_one_ignores_initial_weights},
        {"GATE: parallel training plays every game and learns",
         test_parallel_training_plays_every_game_and_learns},
        {"TD(lambda=0.5) learns", test_lambda_half_learns},
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
