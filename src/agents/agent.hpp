#pragma once

#include "core/board.hpp"

#include <cstdint>
#include <optional>
#include <string_view>

namespace adversarial_2048 {

class Agent {
public:
    virtual ~Agent() = default;

    [[nodiscard]] virtual std::string_view name() const noexcept = 0;
    virtual void reset(std::uint64_t seed) = 0;
    [[nodiscard]] virtual std::optional<Direction> choose_move(Board board) = 0;
};

}  // namespace adversarial_2048
