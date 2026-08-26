#include "learning/temporal_coherence.hpp"

#include <array>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string>

namespace adversarial_2048::learning {

TemporalCoherenceLearner::TemporalCoherenceLearner(const NTupleNetwork& network)
    : error_sum_(network.total_weight_count(), 0.0F),
      absolute_error_sum_(network.total_weight_count(), 0.0F) {
    scratch_.reserve(network.active_weight_count());
}

double TemporalCoherenceLearner::update(
    NTupleNetwork& network, Board afterstate, double target, double alpha) {
    network.active_indices(afterstate, scratch_);
    if (scratch_.empty()) {
        return 0.0;
    }

    auto& weights = network.weights();

    // V(afterstate) summed from the same indices used for the write below.
    // Accumulation order matches NTupleNetwork::value() exactly -- tuple by
    // tuple, ordering by ordering -- so this is bit-identical to calling it,
    // not merely close.
    double current = 0.0;
    for (const auto index : scratch_) {
        current += static_cast<double>(weights[index]);
    }
    const auto delta = target - current;

    // Same 1/m scaling as plain TD, so `alpha` keeps its usual meaning and a
    // TC run is directly comparable to a plain run at the same alpha.
    const auto step = alpha * delta / static_cast<double>(scratch_.size());
    const auto magnitude = static_cast<float>(std::abs(delta));
    const auto signed_delta = static_cast<float>(delta);

    double beta_total = 0.0;
    for (const auto index : scratch_) {
        const auto absolute = absolute_error_sum_[index];
        // A weight with no history has no evidence of oscillation yet, so it
        // gets a full step -- this makes TC reduce exactly to plain TD on the
        // first visit to any weight.
        const auto beta = absolute > 0.0F
            ? static_cast<double>(std::abs(error_sum_[index])) / static_cast<double>(absolute)
            : 1.0;

        weights[index] += static_cast<float>(beta * step);

        // Accumulate AFTER stepping, so beta reflects history strictly before
        // this update. Note an index can repeat within one board when two
        // symmetric orderings collide; treating each occurrence as its own
        // update is what keeps this consistent with NTupleNetwork::update.
        error_sum_[index] += signed_delta;
        absolute_error_sum_[index] += magnitude;
        beta_total += beta;
    }
    return beta_total / static_cast<double>(scratch_.size());
}

double TemporalCoherenceLearner::mean_beta() const {
    double total = 0.0;
    std::size_t visited = 0;
    for (std::size_t index = 0; index < absolute_error_sum_.size(); ++index) {
        const auto absolute = absolute_error_sum_[index];
        if (absolute > 0.0F) {
            total += static_cast<double>(std::abs(error_sum_[index])) /
                     static_cast<double>(absolute);
            ++visited;
        }
    }
    return visited == 0 ? 0.0 : total / static_cast<double>(visited);
}

namespace {

constexpr char kTcMagic[8] = {'A', '2', '0', '4', '8', 'T', 'C', 'S'};

}  // namespace

void TemporalCoherenceLearner::save(const std::filesystem::path& path) const {
    auto temporary = path;
    temporary += ".tmp";
    {
        std::ofstream stream(temporary, std::ios::binary | std::ios::trunc);
        if (!stream) {
            throw std::runtime_error("cannot open TC state for writing: " + temporary.string());
        }
        stream.write(kTcMagic, sizeof(kTcMagic));
        const auto count = static_cast<std::uint64_t>(error_sum_.size());
        stream.write(reinterpret_cast<const char*>(&count), sizeof(count));
        stream.write(reinterpret_cast<const char*>(error_sum_.data()),
                     static_cast<std::streamsize>(error_sum_.size() * sizeof(float)));
        stream.write(reinterpret_cast<const char*>(absolute_error_sum_.data()),
                     static_cast<std::streamsize>(absolute_error_sum_.size() * sizeof(float)));
        if (!stream) {
            throw std::runtime_error("failed writing TC state: " + temporary.string());
        }
    }
    std::filesystem::rename(temporary, path);
}

void TemporalCoherenceLearner::load(const std::filesystem::path& path) {
    std::ifstream stream(path, std::ios::binary);
    if (!stream) {
        throw std::runtime_error("cannot open TC state: " + path.string());
    }
    std::array<char, sizeof(kTcMagic)> magic{};
    stream.read(magic.data(), magic.size());
    if (!stream || std::memcmp(magic.data(), kTcMagic, sizeof(kTcMagic)) != 0) {
        throw std::runtime_error("not a TC state file: " + path.string());
    }
    std::uint64_t count = 0;
    stream.read(reinterpret_cast<char*>(&count), sizeof(count));
    // Size mismatch means this state belongs to a different network shape;
    // silently accepting it would corrupt every step size.
    if (!stream || count != error_sum_.size()) {
        throw std::runtime_error("TC state has " + std::to_string(count) +
                                 " entries but this network needs " +
                                 std::to_string(error_sum_.size()) + ": " + path.string());
    }
    stream.read(reinterpret_cast<char*>(error_sum_.data()),
                static_cast<std::streamsize>(error_sum_.size() * sizeof(float)));
    stream.read(reinterpret_cast<char*>(absolute_error_sum_.data()),
                static_cast<std::streamsize>(absolute_error_sum_.size() * sizeof(float)));
    if (!stream) {
        throw std::runtime_error("truncated TC state: " + path.string());
    }
}

void apply_optimistic_initialisation(NTupleNetwork& network, double initial_value) {
    if (initial_value == 0.0) {
        return;
    }
    const auto per_weight = static_cast<float>(
        initial_value / static_cast<double>(network.active_weight_count()));
    for (auto& weight : network.weights()) {
        weight = per_weight;
    }
}

}  // namespace adversarial_2048::learning
