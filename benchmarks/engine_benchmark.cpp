#include "core/board.hpp"
#include "core/move_tables.hpp"

#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <iostream>

namespace a2048 = adversarial_2048;

namespace {

[[nodiscard]] std::uint64_t next_random(std::uint64_t& state) noexcept {
    state ^= state << 13U;
    state ^= state >> 7U;
    state ^= state << 17U;
    return state;
}

}  // namespace

int main() {
    constexpr std::size_t board_count = 4'096;
    constexpr std::size_t move_count = 20'000'000;
    std::array<a2048::Board, board_count> boards{};
    std::uint64_t random_state = 0x2048C0FFEEULL;

    for (auto& board : boards) {
        a2048::CellArray cells{};
        for (auto& exponent : cells) {
            const auto sample = next_random(random_state);
            exponent = sample % 5U == 0U ? 0U : static_cast<std::uint8_t>(1U + sample % 10U);
        }
        board = a2048::encode(cells);
    }

    // Initialize the tables before timing the hot move path.
    static_cast<void>(a2048::move_tables());

    std::uint64_t checksum = 0;
    std::uint64_t total_score = 0;
    const auto start = std::chrono::steady_clock::now();
    for (std::size_t iteration = 0; iteration < move_count; ++iteration) {
        const auto direction = static_cast<a2048::Direction>(iteration & 3U);
        const auto result = a2048::move(boards[iteration & (board_count - 1U)], direction);
        checksum ^= result.board.packed_exponents + iteration;
        checksum ^= static_cast<std::uint64_t>(result.board.exponent_high_bits) << 32U;
        total_score += result.score;
    }
    const auto elapsed = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - start).count();

    std::cout << std::fixed << std::setprecision(2)
              << "moves:       " << move_count << '\n'
              << "seconds:     " << elapsed << '\n'
              << "moves/sec:   " << static_cast<double>(move_count) / elapsed << '\n'
              << "ns/move:     " << elapsed * 1'000'000'000.0 /
                                         static_cast<double>(move_count) << '\n'
              << "checksum:    " << checksum << '\n'
              << "score total: " << total_score << '\n';
}
