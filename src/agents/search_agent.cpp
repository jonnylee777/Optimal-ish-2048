#include "agents/search_agent.hpp"

#include "core/board.hpp"

#include <algorithm>
#include <stdexcept>
#include <utility>

namespace adversarial_2048 {

std::uint32_t AdaptiveDepthSchedule::select(Board board) const noexcept {
    const auto empties = empty_count(board);
    if (empties >= 10U) {
        return high_empty_depth;
    }
    if (empties >= 6U) {
        return medium_empty_depth;
    }
    return low_empty_depth;
}

std::uint32_t AdaptiveDepthSchedule::maximum_depth() const noexcept {
    return std::max({high_empty_depth, medium_empty_depth, low_empty_depth});
}

SearchAgent::SearchAgent(
    const Evaluator& evaluator,
    std::uint32_t depth,
    std::string name,
    ExpectimaxOptions options)
    : name_(std::move(name)),
      depth_(depth),
      minimum_path_probability_(options.minimum_path_probability),
      time_limit_seconds_(options.time_limit_seconds),
      search_(evaluator, options) {
    if (depth_ == 0) {
        throw std::invalid_argument("search agent depth must be at least one");
    }
}

SearchAgent::SearchAgent(
    const Evaluator& evaluator,
    AdaptiveDepthSchedule schedule,
    std::string name,
    ExpectimaxOptions options)
    : name_(std::move(name)),
      depth_(schedule.maximum_depth()),
      adaptive_schedule_(schedule),
      adaptive_depth_(true),
      minimum_path_probability_(options.minimum_path_probability),
      time_limit_seconds_(options.time_limit_seconds),
      search_(evaluator, options) {
    if (schedule.high_empty_depth == 0 || schedule.medium_empty_depth == 0 ||
        schedule.low_empty_depth == 0) {
        throw std::invalid_argument("adaptive search depths must be at least one");
    }
}

std::string_view SearchAgent::name() const noexcept {
    return name_;
}

void SearchAgent::reset(std::uint64_t) {}

std::optional<Direction> SearchAgent::choose_move(Board board) {
    auto selected_depth = depth_;
    if (adaptive_depth_) {
        const auto empties = empty_count(board);
        const auto bucket = empties >= 10U ? 0U : (empties >= 6U ? 1U : 2U);
        ++adaptive_depth_usage_[bucket];
        selected_depth = adaptive_schedule_.select(board);
    }
    const auto result = time_limit_seconds_ > 0.0
        ? search_.search_iterative(board, selected_depth)
        : search_.search(board, selected_depth);
    if (result.completed_depth < completed_depth_usage_.size()) {
        ++completed_depth_usage_[result.completed_depth];
    }
    cumulative_.player_nodes += result.statistics.player_nodes;
    cumulative_.chance_nodes += result.statistics.chance_nodes;
    cumulative_.leaf_evaluations += result.statistics.leaf_evaluations;
    cumulative_.spawn_outcomes += result.statistics.spawn_outcomes;
    cumulative_.cache_lookups += result.statistics.cache_lookups;
    cumulative_.cache_hits += result.statistics.cache_hits;
    cumulative_.elapsed_seconds += result.statistics.elapsed_seconds;
    return result.direction;
}

bool SearchAgent::uses_adaptive_depth() const noexcept {
    return adaptive_depth_;
}

const std::array<std::uint64_t, 3>& SearchAgent::adaptive_depth_usage() const noexcept {
    return adaptive_depth_usage_;
}

const std::array<std::uint64_t, 13>& SearchAgent::completed_depth_usage() const noexcept {
    return completed_depth_usage_;
}

std::uint32_t SearchAgent::depth() const noexcept {
    return depth_;
}

double SearchAgent::minimum_path_probability() const noexcept {
    return minimum_path_probability_;
}

double SearchAgent::time_limit_seconds() const noexcept {
    return time_limit_seconds_;
}

const SearchStatistics& SearchAgent::cumulative_statistics() const noexcept {
    return cumulative_;
}

void SearchAgent::clear_statistics() noexcept {
    cumulative_ = {};
    adaptive_depth_usage_ = {};
    completed_depth_usage_ = {};
}

}  // namespace adversarial_2048
