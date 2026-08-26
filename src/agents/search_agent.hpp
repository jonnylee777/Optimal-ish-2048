#pragma once

#include "agents/agent.hpp"
#include "evaluation/evaluator.hpp"
#include "search/expectimax.hpp"

#include <cstdint>
#include <array>
#include <string>

namespace adversarial_2048 {

struct AdaptiveDepthSchedule {
    std::uint32_t high_empty_depth{4};
    std::uint32_t medium_empty_depth{6};
    std::uint32_t low_empty_depth{8};

    [[nodiscard]] std::uint32_t select(Board board) const noexcept;
    [[nodiscard]] std::uint32_t maximum_depth() const noexcept;
};

class SearchAgent final : public Agent {
public:
    SearchAgent(
        const Evaluator& evaluator,
        std::uint32_t depth,
        std::string name = "expectimax",
        ExpectimaxOptions options = {});
    SearchAgent(
        const Evaluator& evaluator,
        AdaptiveDepthSchedule schedule,
        std::string name,
        ExpectimaxOptions options = {});

    [[nodiscard]] std::string_view name() const noexcept override;
    void reset(std::uint64_t seed) override;
    [[nodiscard]] std::optional<Direction> choose_move(Board board) override;

    [[nodiscard]] std::uint32_t depth() const noexcept;
    [[nodiscard]] bool uses_adaptive_depth() const noexcept;
    [[nodiscard]] const std::array<std::uint64_t, 3>& adaptive_depth_usage() const noexcept;
    [[nodiscard]] const std::array<std::uint64_t, 13>& completed_depth_usage() const noexcept;
    [[nodiscard]] double minimum_path_probability() const noexcept;
    [[nodiscard]] double time_limit_seconds() const noexcept;
    [[nodiscard]] const SearchStatistics& cumulative_statistics() const noexcept;
    void clear_statistics() noexcept;

private:
    std::string name_;
    std::uint32_t depth_;
    AdaptiveDepthSchedule adaptive_schedule_{};
    bool adaptive_depth_{};
    std::array<std::uint64_t, 3> adaptive_depth_usage_{};
    std::array<std::uint64_t, 13> completed_depth_usage_{};
    double minimum_path_probability_;
    double time_limit_seconds_{};
    Expectimax search_;
    SearchStatistics cumulative_{};
};

}  // namespace adversarial_2048
