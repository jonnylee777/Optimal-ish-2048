// Temporal-coherence learning and optimistic initialisation.
//
// The failure mode these guard against is silent: a per-weight step rule that
// compiles, runs, and updates *something* still looks like "training is just
// slow" if beta is computed wrongly. So the tests pin beta's two defining
// limits by hand rather than only checking that scores go up.
#include "core/board.hpp"
#include "learning/ntuple_network.hpp"
#include "learning/td_trainer.hpp"
#include "learning/temporal_coherence.hpp"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <random>
#include <exception>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace a2048 = adversarial_2048;
namespace nn = adversarial_2048::learning;

namespace {

class TestFailure : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

#define CHECK(condition)                                                                     \
    do {                                                                                     \
        if (!(condition)) {                                                                  \
            throw TestFailure(std::string("check failed: ") + #condition + " at line " +     \
                              std::to_string(__LINE__));                                     \
        }                                                                                    \
    } while (false)

#define CHECK_NEAR(a, b, eps)                                                                \
    do {                                                                                     \
        const double lhs_ = (a);                                                             \
        const double rhs_ = (b);                                                             \
        if (std::abs(lhs_ - rhs_) > (eps)) {                                                 \
            throw TestFailure(std::string("expected ") + std::to_string(lhs_) + " ~= " +     \
                              std::to_string(rhs_) + " at line " + std::to_string(__LINE__)); \
        }                                                                                    \
    } while (false)

[[nodiscard]] a2048::Board board_with(
    std::initializer_list<std::pair<std::size_t, std::uint8_t>> tiles) {
    a2048::CellArray cells{};
    for (const auto& [index, exponent] : tiles) {
        cells[index] = exponent;
    }
    return a2048::encode(cells);
}

// A single tuple keeps the arithmetic hand-checkable.
[[nodiscard]] std::vector<nn::TupleSpec> one_tuple() {
    return {nn::TupleSpec{{0, 1, 2, 3}}};
}

// TemporalCoherenceLearner::update takes a TARGET, but these tests are written
// in terms of a controlled sequence of TD ERRORS -- that is what beta is a
// function of. Converting here keeps each test's intent explicit: the delta
// applied is exactly `error`, because target - V(board) == error by
// construction.
double update_with_error(nn::TemporalCoherenceLearner& learner, nn::NTupleNetwork& network,
                         a2048::Board board, double error, double alpha) {
    return learner.update(network, board, network.value(board) + error, alpha);
}

// GATE: the first update must behave exactly like plain TD. A weight with no
// history has no evidence of oscillation, so beta must be 1 -- otherwise TC
// silently shrinks every early update and looks like a bad learning rate.
void test_first_update_matches_plain_td() {
    const auto board = board_with({{0, 1}, {1, 2}, {2, 3}, {3, 4}});
    const double alpha = 0.5;
    const double delta = 8.0;

    nn::NTupleNetwork plain(one_tuple());
    nn::NTupleNetwork coherent(one_tuple());
    nn::TemporalCoherenceLearner learner(coherent);

    plain.update(board, alpha * delta / static_cast<double>(plain.active_weight_count()));
    const auto beta = update_with_error(learner, coherent, board, delta, alpha);

    CHECK_NEAR(beta, 1.0, 1e-12);
    CHECK_NEAR(coherent.value(board), plain.value(board), 1e-9);
}

// GATE: errors that always point the same way must keep beta at 1 (|E| == A).
// This is the "still converging, keep moving" case.
void test_consistent_errors_keep_full_step() {
    const auto board = board_with({{0, 1}, {1, 2}, {2, 3}, {3, 4}});
    nn::NTupleNetwork network(one_tuple());
    nn::TemporalCoherenceLearner learner(network);

    double beta = 0.0;
    for (int iteration = 0; iteration < 10; ++iteration) {
        beta = update_with_error(learner, network, board, +5.0, 0.1);
    }
    CHECK_NEAR(beta, 1.0, 1e-9);
    CHECK_NEAR(learner.mean_beta(), 1.0, 1e-9);
}

// GATE: errors that alternate in sign must drive beta toward 0 (|E| << A).
// This is the "oscillating, damp it" case and the whole point of TC.
void test_alternating_errors_damp_the_step() {
    const auto board = board_with({{0, 1}, {1, 2}, {2, 3}, {3, 4}});
    nn::NTupleNetwork network(one_tuple());
    nn::TemporalCoherenceLearner learner(network);

    double beta = 1.0;
    for (int iteration = 0; iteration < 200; ++iteration) {
        beta = update_with_error(learner, network, board,
                                 (iteration % 2 == 0) ? +5.0 : -5.0, 0.1);
    }
    // E oscillates between 0 and +/-5 while A grows without bound, so beta
    // must decay toward zero rather than merely getting smaller once.
    CHECK(beta < 0.02);
    CHECK(learner.mean_beta() < 0.02);
}

// A damped weight must actually take smaller steps than an undamped one, not
// just report a smaller beta.
void test_damping_reduces_actual_weight_movement() {
    const auto board = board_with({{0, 5}, {1, 6}, {2, 7}, {3, 8}});

    nn::NTupleNetwork consistent(one_tuple());
    nn::TemporalCoherenceLearner consistent_learner(consistent);
    nn::NTupleNetwork oscillating(one_tuple());
    nn::TemporalCoherenceLearner oscillating_learner(oscillating);

    for (int iteration = 0; iteration < 50; ++iteration) {
        update_with_error(consistent_learner, consistent, board, +5.0, 0.1);
        update_with_error(oscillating_learner, oscillating, board,
                          (iteration % 2 == 0) ? +5.0 : -5.0, 0.1);
    }

    const auto before_consistent = consistent.value(board);
    const auto before_oscillating = oscillating.value(board);
    update_with_error(consistent_learner, consistent, board, +5.0, 0.1);
    update_with_error(oscillating_learner, oscillating, board, +5.0, 0.1);

    const auto moved_consistent = std::abs(consistent.value(board) - before_consistent);
    const auto moved_oscillating = std::abs(oscillating.value(board) - before_oscillating);
    CHECK(moved_oscillating < moved_consistent * 0.1);
}

// Optimistic initialisation must make an untouched board evaluate to the
// requested value -- that is the entire contract, since greedy selection is
// what consumes it.
void test_optimistic_initialisation_hits_target_value() {
    nn::NTupleNetwork network(nn::default_tuple_specs());
    nn::apply_optimistic_initialisation(network, 320000.0);

    // Any board, visited or not, reads back the target before any learning.
    CHECK_NEAR(network.value(board_with({{0, 1}, {5, 3}})), 320000.0, 1.0);
    CHECK_NEAR(network.value(board_with({{15, 11}})), 320000.0, 1.0);
}

void test_optimistic_initialisation_zero_is_noop() {
    nn::NTupleNetwork network(one_tuple());
    const auto board = board_with({{0, 1}, {1, 2}});
    network.update(board, 3.0);
    const auto before = network.value(board);
    nn::apply_optimistic_initialisation(network, 0.0);
    CHECK_NEAR(network.value(board), before, 1e-12);
}

// active_indices() must agree with update() about which weights a board
// touches, including repeated indices from colliding symmetric orderings --
// TC's bookkeeping is only correct if the two never disagree.
void test_active_indices_agree_with_update() {
    nn::NTupleNetwork probe(nn::default_tuple_specs());
    nn::NTupleNetwork direct(nn::default_tuple_specs());
    // A deliberately symmetric board, where distinct orderings collide.
    const auto board = board_with({{0, 3}, {3, 3}, {12, 3}, {15, 3}});

    std::vector<std::size_t> indices;
    probe.active_indices(board, indices);
    CHECK(indices.size() == probe.active_weight_count());

    const double step = 0.25;
    for (const auto index : indices) {
        probe.weights()[index] += static_cast<float>(step);
    }
    direct.update(board, step);
    CHECK_NEAR(probe.value(board), direct.value(board), 1e-6);
}

// GATE: summing the weights at active_indices() must equal value() EXACTLY,
// not approximately.
//
// TemporalCoherenceLearner::update relies on this: it computes V(afterstate)
// from the indices it already needs for the write, instead of calling value()
// separately. That is only a valid optimisation if the two agree bit-for-bit --
// a last-ulp difference would silently perturb every TD error and make TC runs
// irreproducible against the plain path.
void test_index_sum_equals_value_exactly() {
    nn::NTupleNetwork network(nn::default_tuple_specs());
    std::mt19937_64 rng(0x7C0FFEE);

    std::vector<std::size_t> indices;
    for (int trial = 0; trial < 200; ++trial) {
        // Random board, and random weights so the sum is not trivially zero.
        a2048::CellArray cells{};
        for (auto& cell : cells) {
            cell = static_cast<std::uint8_t>(rng() % 16);
        }
        const auto board = a2048::encode(cells);
        network.update(board, static_cast<double>(rng() % 1000) / 7.0);

        network.active_indices(board, indices);
        double summed = 0.0;
        for (const auto index : indices) {
            summed += static_cast<double>(network.weights()[index]);
        }
        // Exact equality, deliberately not CHECK_NEAR.
        CHECK(summed == network.value(board));
    }
}

// GATE: TC state must round-trip, because that is what makes long training
// incremental. If the accumulators came back wrong, every subsequent step size
// would be wrong and a converged network would be damaged rather than improved.
void test_tc_state_round_trips() {
    const auto board = board_with({{0, 3}, {1, 4}, {2, 5}, {3, 6}});
    nn::NTupleNetwork network(one_tuple());
    nn::TemporalCoherenceLearner original(network);
    // Build up an asymmetric history so beta is neither 0 nor 1.
    for (int i = 0; i < 20; ++i) {
        update_with_error(original, network, board, (i % 3 == 0) ? -4.0 : +6.0, 0.1);
    }
    const auto expected_beta = original.mean_beta();
    CHECK(expected_beta > 0.0);
    CHECK(expected_beta < 1.0);

    const auto path = std::filesystem::temp_directory_path() / "a2048_tc_state_test.bin";
    original.save(path);

    nn::NTupleNetwork fresh(one_tuple());
    nn::TemporalCoherenceLearner restored(fresh);
    // A fresh learner has no history, so its beta is 1.0 by construction.
    CHECK(restored.mean_beta() == 0.0);
    restored.load(path);
    CHECK(std::abs(restored.mean_beta() - expected_beta) < 1e-9);

    // And the restored learner must produce the same next step as the original.
    nn::NTupleNetwork a(one_tuple());
    nn::NTupleNetwork b(one_tuple());
    const auto beta_a = update_with_error(original, a, board, +5.0, 0.1);
    const auto beta_b = update_with_error(restored, b, board, +5.0, 0.1);
    CHECK(std::abs(beta_a - beta_b) < 1e-12);
    std::filesystem::remove(path);
}

// A state file from a different-sized network must be rejected, not silently
// applied to the wrong weights.
void test_tc_state_rejects_size_mismatch() {
    nn::NTupleNetwork small(one_tuple());
    nn::TemporalCoherenceLearner small_learner(small);
    const auto path = std::filesystem::temp_directory_path() / "a2048_tc_mismatch.bin";
    small_learner.save(path);

    nn::NTupleNetwork big(nn::default_tuple_specs());
    nn::TemporalCoherenceLearner big_learner(big);
    bool threw = false;
    try {
        big_learner.load(path);
    } catch (const std::exception&) {
        threw = true;
    }
    std::filesystem::remove(path);
    CHECK(threw);
}

// End-to-end: TC training must learn. A run that produces a network no better
// than an untrained one would pass every unit test above.
void test_tc_training_learns() {
    // Calibrate against an UNTRAINED network rather than a hardcoded score.
    //
    // Note what that baseline actually is: with all weights zero, V is zero
    // everywhere, so argmax(reward + V) degenerates to "take the largest
    // immediate merge". That is a genuine greedy heuristic scoring ~3,450 --
    // NOT random play. Any threshold picked without measuring it would be
    // meaningless, and "beats random" would be a trivially passing test.
    const auto greedy_mean = [](const nn::NTupleNetwork& net) {
        constexpr int kGames = 40;
        double total = 0.0;
        for (int game = 0; game < kGames; ++game) {
            total += static_cast<double>(
                nn::play_greedy_game(net, 0xBA5E0000ULL + static_cast<std::uint64_t>(game))
                    .score);
        }
        return total / kGames;
    };

    nn::NTupleNetwork untrained(nn::default_tuple_specs());
    const auto baseline = greedy_mean(untrained);

    nn::NTupleNetwork network(nn::default_tuple_specs());
    nn::TrainConfig config;
    config.games = 2000;
    config.alpha = 0.1;
    config.seed = 7;
    config.temporal_coherence = true;
    const auto result = nn::train(network, config);

    CHECK(result.games_played == 2000);
    // Compare the FINAL network's greedy play, not result.mean_score -- the
    // latter averages over the whole run including the untrained early games,
    // so it understates what was learned.
    CHECK(greedy_mean(network) > baseline * 1.5);

    // Beta must be a real ratio that has actually moved off its 1.0 start --
    // a beta pinned at 1.0 would mean TC never damped anything and the run
    // was plain TD wearing a costume.
    CHECK(result.final_mean_beta > 0.0);
    CHECK(result.final_mean_beta < 1.0);
}

struct NamedTest {
    const char* name;
    void (*function)();
};

const NamedTest kTests[] = {
    {"GATE: first TC update equals plain TD", test_first_update_matches_plain_td},
    {"GATE: consistent errors keep beta at 1", test_consistent_errors_keep_full_step},
    {"GATE: alternating errors drive beta to 0", test_alternating_errors_damp_the_step},
    {"damping shrinks real weight movement", test_damping_reduces_actual_weight_movement},
    {"optimistic init hits its target value", test_optimistic_initialisation_hits_target_value},
    {"optimistic init with 0 is a no-op", test_optimistic_initialisation_zero_is_noop},
    {"active_indices agrees with update", test_active_indices_agree_with_update},
    {"GATE: index sum equals value() exactly", test_index_sum_equals_value_exactly},
    {"GATE: TC state round-trips", test_tc_state_round_trips},
    {"TC state rejects size mismatch", test_tc_state_rejects_size_mismatch},
    {"TC training actually learns", test_tc_training_learns},
};

}  // namespace

int main() {
    int failures = 0;
    for (const auto& test : kTests) {
        try {
            test.function();
            std::cout << "[PASS] " << test.name << '\n';
        } catch (const std::exception& error) {
            std::cout << "[FAIL] " << test.name << ": " << error.what() << '\n';
            ++failures;
        }
    }
    const auto total = sizeof(kTests) / sizeof(kTests[0]);
    std::cout << (total - static_cast<std::size_t>(failures)) << '/' << total
              << " tests passed\n";
    return failures == 0 ? 0 : 1;
}
