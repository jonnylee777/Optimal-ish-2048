#pragma once

#include "agents/agent.hpp"

namespace adversarial_2048 {

// Chooses the legal move with the largest immediate merge score. Equal scores
// use the stable kDirections order so experiments are exactly reproducible.
class GreedyAgent final : public Agent {
public:
    [[nodiscard]] std::string_view name() const noexcept override;
    void reset(std::uint64_t seed) override;
    [[nodiscard]] std::optional<Direction> choose_move(Board board) override;
};

}  // namespace adversarial_2048
