#include "core/spawn.hpp"

namespace adversarial_2048 {

std::optional<SpawnEvent> spawn_random(Board& board, RandomEngine& rng) {
    const auto empties = empty_count(board);
    if (empties == 0) {
        return std::nullopt;
    }

    auto selected_empty = static_cast<std::size_t>(sample_bounded(rng, empties));
    std::size_t selected_index = 0;
    for (; selected_index < kCellCount; ++selected_index) {
        if (cell_at(board, selected_index) == 0) {
            if (selected_empty == 0) {
                break;
            }
            --selected_empty;
        }
    }

    const auto exponent = static_cast<std::uint8_t>(sample_bounded(rng, 10) == 0 ? 2 : 1);
    board = with_cell(board, selected_index, exponent);
    return SpawnEvent{selected_index, exponent};
}

}  // namespace adversarial_2048
