#pragma once

#include "experiments/game_runner.hpp"

#include <cstdint>
#include <string_view>

namespace adversarial_2048::seed_sets {

inline constexpr RunConfig training{1'000, 1'000};
inline constexpr RunConfig validation{500, 10'000};
inline constexpr RunConfig final_test{1'000, 50'000};

// Standardized cross-heuristic comparison sets (see docs/phase1-heuristics.md).
// Distinct purpose from training/validation/final_test above, which exist to
// keep weight optimization from overfitting to its own selection seeds.
// Disjoint from those ranges and from each other by construction.
inline constexpr RunConfig quick_benchmark{100, 20'000};
inline constexpr RunConfig standard_benchmark{500, 30'000};
inline constexpr RunConfig final_benchmark{2'000, 40'000};

[[nodiscard]] inline std::string_view classify(
    std::uint64_t first_seed,
    std::uint64_t last_seed) noexcept {
    const auto within = [first_seed, last_seed](RunConfig set) {
        const auto set_last = set.first_seed + set.game_count - 1U;
        return first_seed >= set.first_seed && last_seed <= set_last;
    };
    if (within(training)) {
        return "training";
    }
    if (within(validation)) {
        return "validation";
    }
    if (within(final_test)) {
        return "final-test";
    }
    if (within(quick_benchmark)) {
        return "quick-benchmark";
    }
    if (within(standard_benchmark)) {
        return "standard-benchmark";
    }
    if (within(final_benchmark)) {
        return "final-benchmark";
    }
    return "custom";
}

}  // namespace adversarial_2048::seed_sets
