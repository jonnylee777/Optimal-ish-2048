#pragma once

#include "core/board.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

namespace adversarial_2048::learning {

// A small neural value function, as an alternative to the n-tuple lookup table.
//
// WHY. The n-tuple network is a table over exact tile values in fixed groups of
// squares. It therefore cannot represent a relational fact — "the big tiles are
// in descending order along an edge" — no matter how many weights it is given.
// That is the concept deciding these games: a hand-written formula encoding it
// reaches 109,213 with no learning at all, while eight separate attempts to
// improve the learned table have all landed at ~350,000. A network can compose
// features across the whole board; a table cannot.
//
// ARCHITECTURE. Deliberately small, because search calls this millions of times.
//
//     board -> 16 one-hot cells (cell * 16 + exponent, so 256 possible rows)
//           -> sum the 16 corresponding embedding rows      [hidden]
//           -> ReLU
//           -> linear                                       [scalar value]
//
// The input is SPARSE — exactly 16 of 256 entries are non-zero — so the first
// layer is 16 contiguous row-additions rather than a 256xH matrix multiply.
// At hidden=256 that is ~4,400 multiply-adds over 257 KB of weights, which sits
// entirely in L2. The n-tuple it replaces does 40 scattered reads into 320 MB,
// every one a cache miss, so this is competitive on speed and may be faster.
//
// HONEST RISK. n-tuple networks beat neural networks in most published 2048
// work, because raw memorisation capacity matters more than generalisation in a
// state space this large. This model has ~66k weights against the table's 83.9
// million. It may simply underfit. That is the experiment.
class NeuralValueNetwork {
public:
    explicit NeuralValueNetwork(std::size_t hidden_units = 256, std::uint64_t seed = 1);

    [[nodiscard]] double value(Board board) const noexcept;

    // Applies one gradient step that moves value(board) toward value + delta.
    // `scale` is the learning rate; the caller supplies the TD error as delta,
    // exactly as the n-tuple trainer does.
    void update(Board board, double delta, double scale) noexcept;

    [[nodiscard]] std::size_t hidden_units() const noexcept { return hidden_; }
    [[nodiscard]] std::size_t weight_count() const noexcept;

    void save(const std::filesystem::path& path) const;
    void load(const std::filesystem::path& path);
    [[nodiscard]] std::string fingerprint() const;

private:
    // Fills `out` with the 16 embedding-row indices this board activates.
    static void active_rows(Board board, std::array<std::size_t, kCellCount>& out) noexcept;

    std::size_t hidden_{};
    std::vector<float> embedding_;   // [256 * hidden]
    std::vector<float> hidden_bias_; // [hidden]
    std::vector<float> output_;      // [hidden]
    float output_bias_{};

    // Scratch reused across calls so the hot path allocates nothing.
    mutable std::vector<float> pre_activation_;
};

}  // namespace adversarial_2048::learning
