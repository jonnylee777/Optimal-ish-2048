#pragma once

#include <cstdint>
#include <random>
#include <stdexcept>

namespace adversarial_2048 {

using RandomEngine = std::mt19937_64;

// Samples [0, bound) without modulo bias. Using the standardized engine output
// directly keeps seeded experiments reproducible across standard libraries.
[[nodiscard]] inline std::uint64_t sample_bounded(
    RandomEngine& rng,
    std::uint64_t bound) {
    if (bound == 0) {
        throw std::invalid_argument("random sample bound must be positive");
    }

    const auto rejection_threshold = static_cast<std::uint64_t>(-bound) % bound;
    while (true) {
        const auto sample = rng();
        if (sample >= rejection_threshold) {
            return sample % bound;
        }
    }
}

}  // namespace adversarial_2048
