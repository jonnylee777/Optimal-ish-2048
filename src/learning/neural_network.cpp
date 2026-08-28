#include "learning/neural_network.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <random>
#include <sstream>
#include <stdexcept>

namespace adversarial_2048::learning {
namespace {

constexpr std::size_t kRows = 256;  // 16 cells x 16 exponent values
constexpr char kMagic[8] = {'A', '2', '0', '4', '8', 'N', 'N', 'V'};

}  // namespace

NeuralValueNetwork::NeuralValueNetwork(std::size_t hidden_units, std::uint64_t seed)
    : hidden_(hidden_units),
      embedding_(kRows * hidden_units, 0.0F),
      hidden_bias_(hidden_units, 0.0F),
      output_(hidden_units, 0.0F),
      pre_activation_(hidden_units, 0.0F) {
    if (hidden_units == 0) {
        throw std::invalid_argument("neural network needs at least one hidden unit");
    }
    // Small random init. Zero would leave every hidden unit identical and their
    // gradients identical too, so the layer could never differentiate — the
    // classic symmetry-breaking requirement. Scaled by 1/sqrt(fan-in) where
    // fan-in is the 16 rows actually summed, not all 256.
    std::mt19937_64 rng(seed);
    std::normal_distribution<float> normal(0.0F, 1.0F / std::sqrt(16.0F));
    for (auto& weight : embedding_) {
        weight = normal(rng);
    }
    std::normal_distribution<float> output_normal(
        0.0F, 1.0F / std::sqrt(static_cast<float>(hidden_units)));
    for (auto& weight : output_) {
        weight = output_normal(rng);
    }
}

void NeuralValueNetwork::active_rows(
    Board board, std::array<std::size_t, kCellCount>& out) noexcept {
    const auto cells = decode(board);
    for (std::size_t cell = 0; cell < kCellCount; ++cell) {
        auto exponent = static_cast<std::size_t>(cells[cell]);
        if (exponent > 15) {
            exponent = 15;  // same clamp the n-tuple uses for 65536+
        }
        out[cell] = cell * 16U + exponent;
    }
}

double NeuralValueNetwork::value(Board board) const noexcept {
    std::array<std::size_t, kCellCount> rows{};
    active_rows(board, rows);

    // Layer 1: sum 16 embedding rows. Sparse input means this is row additions,
    // not a matrix multiply.
    std::copy(hidden_bias_.begin(), hidden_bias_.end(), pre_activation_.begin());
    for (const auto row : rows) {
        const auto* source = embedding_.data() + row * hidden_;
        for (std::size_t unit = 0; unit < hidden_; ++unit) {
            pre_activation_[unit] += source[unit];
        }
    }

    // ReLU, then the linear read-out, fused into one pass.
    double total = static_cast<double>(output_bias_);
    for (std::size_t unit = 0; unit < hidden_; ++unit) {
        if (pre_activation_[unit] > 0.0F) {
            total += static_cast<double>(output_[unit]) *
                     static_cast<double>(pre_activation_[unit]);
        }
    }
    return total;
}

void NeuralValueNetwork::update(Board board, double delta, double scale) noexcept {
    std::array<std::size_t, kCellCount> rows{};
    active_rows(board, rows);

    std::copy(hidden_bias_.begin(), hidden_bias_.end(), pre_activation_.begin());
    for (const auto row : rows) {
        const auto* source = embedding_.data() + row * hidden_;
        for (std::size_t unit = 0; unit < hidden_; ++unit) {
            pre_activation_[unit] += source[unit];
        }
    }

    const auto step = static_cast<float>(scale * delta);

    // Output layer: d(value)/d(output_[j]) = activation_j.
    // Hidden layer:  d(value)/d(pre_j)     = output_[j], gated by the ReLU.
    //
    // The output weights are read BEFORE being modified, so the hidden gradient
    // uses the pre-update values — updating in the other order would apply a
    // subtly wrong gradient that still looks like learning.
    for (std::size_t unit = 0; unit < hidden_; ++unit) {
        const auto pre = pre_activation_[unit];
        if (pre <= 0.0F) {
            continue;  // ReLU is flat here: no gradient flows to this unit
        }
        const auto output_weight = output_[unit];
        output_[unit] += step * pre;
        const auto hidden_gradient = step * output_weight;
        hidden_bias_[unit] += hidden_gradient;
        for (const auto row : rows) {
            embedding_[row * hidden_ + unit] += hidden_gradient;
        }
    }
    output_bias_ += step;
}

std::size_t NeuralValueNetwork::weight_count() const noexcept {
    return embedding_.size() + hidden_bias_.size() + output_.size() + 1U;
}

void NeuralValueNetwork::save(const std::filesystem::path& path) const {
    auto temporary = path;
    temporary += ".tmp";
    {
        std::ofstream stream(temporary, std::ios::binary | std::ios::trunc);
        if (!stream) {
            throw std::runtime_error("cannot write neural weights: " + temporary.string());
        }
        stream.write(kMagic, sizeof(kMagic));
        const auto hidden = static_cast<std::uint64_t>(hidden_);
        stream.write(reinterpret_cast<const char*>(&hidden), sizeof(hidden));
        const auto write = [&stream](const std::vector<float>& data) {
            stream.write(reinterpret_cast<const char*>(data.data()),
                         static_cast<std::streamsize>(data.size() * sizeof(float)));
        };
        write(embedding_);
        write(hidden_bias_);
        write(output_);
        stream.write(reinterpret_cast<const char*>(&output_bias_), sizeof(output_bias_));
        if (!stream) {
            throw std::runtime_error("failed writing neural weights: " + temporary.string());
        }
    }
    std::filesystem::rename(temporary, path);
}

void NeuralValueNetwork::load(const std::filesystem::path& path) {
    std::ifstream stream(path, std::ios::binary);
    if (!stream) {
        throw std::runtime_error("cannot open neural weights: " + path.string());
    }
    char magic[sizeof(kMagic)]{};
    stream.read(magic, sizeof(magic));
    if (!stream || std::memcmp(magic, kMagic, sizeof(kMagic)) != 0) {
        throw std::runtime_error("not a neural weight file: " + path.string());
    }
    std::uint64_t hidden = 0;
    stream.read(reinterpret_cast<char*>(&hidden), sizeof(hidden));
    if (!stream || hidden != hidden_) {
        throw std::runtime_error("neural weight file has " + std::to_string(hidden) +
                                 " hidden units but this network has " +
                                 std::to_string(hidden_) + ": " + path.string());
    }
    const auto read = [&stream, &path](std::vector<float>& data) {
        stream.read(reinterpret_cast<char*>(data.data()),
                    static_cast<std::streamsize>(data.size() * sizeof(float)));
        if (!stream) {
            throw std::runtime_error("truncated neural weight file: " + path.string());
        }
    };
    read(embedding_);
    read(hidden_bias_);
    read(output_);
    stream.read(reinterpret_cast<char*>(&output_bias_), sizeof(output_bias_));
    if (!stream) {
        throw std::runtime_error("truncated neural weight file: " + path.string());
    }
}

std::string NeuralValueNetwork::fingerprint() const {
    std::uint64_t hash = 0xCBF29CE484222325ULL;
    const auto absorb = [&hash](const void* data, std::size_t bytes) {
        const auto* raw = static_cast<const unsigned char*>(data);
        for (std::size_t index = 0; index < bytes; ++index) {
            hash ^= raw[index];
            hash *= 0x100000001B3ULL;
        }
    };
    absorb(embedding_.data(), embedding_.size() * sizeof(float));
    absorb(hidden_bias_.data(), hidden_bias_.size() * sizeof(float));
    absorb(output_.data(), output_.size() * sizeof(float));
    absorb(&output_bias_, sizeof(output_bias_));

    std::ostringstream out;
    out << "neural:hidden=" << hidden_ << ",weights=" << weight_count()
        << ",hash=" << std::hex << hash;
    return out.str();
}

}  // namespace adversarial_2048::learning
