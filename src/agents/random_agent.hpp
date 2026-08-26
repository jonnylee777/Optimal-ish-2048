#pragma once

#include "agents/agent.hpp"
#include "core/random.hpp"

namespace adversarial_2048 {

class RandomAgent final : public Agent {
public:
    explicit RandomAgent(std::uint64_t seed = 0);

    [[nodiscard]] std::string_view name() const noexcept override;
    void reset(std::uint64_t seed) override;
    [[nodiscard]] std::optional<Direction> choose_move(Board board) override;

private:
    RandomEngine rng_;
};

}  // namespace adversarial_2048
