// Measure the conversion rate directly, instead of inferring it from whole games.
//
// E31 localised where the 32768 ceiling actually bites: 83.6% of games reach
// 16384 + 8192 -- one merge short of a second 16384 -- and fail to convert. So
// the quantity that decides everything is a single conditional,
//
//     P(second 16384 | 16384 + 8192 on the board)
//
// and whole-game benchmarking is a poor way to measure it. A depth-8 game runs
// ~20,000 moves and only the last ~2,800 test the conditional; the rest is
// expensive noise. This collects the junctures once from a cheap depth-4 run,
// then replays only the tail under whatever configuration is being tested, with
// the spawn stream derived from each juncture's own seed so every arm gets
// MATCHED trials rather than merely comparable ones.
//
// Validated against whole games: 11/224 = 4.91% here, versus 4/127 = 3.1% for
// the same quantity measured from full games in E31.
//
//   convert_probe collect <weights> <out.bin> <games> <first-seed> <threads>
//   convert_probe probe   <weights> <in.bin>  <depth|H,M,L> <cutoff> <threads> [limit]
//
// Set CONVERT_PROBE_DUMP=<path> to write a per-trial CSV (arrival features plus
// outcome), so results can be related to the state the agent arrived in.
//
// COST SCALES WITH NODES PER MOVE, NOT WITH THE DEPTH NUMBER. Measured, per
// move: depth 4 / cutoff 0.0015 = 16.5k nodes; depth 6 / 0.0015 = 103k; depth 6
// / 0.0002 = 523k. The last is 32x the first, so a probe taking 3 minutes at
// baseline settings takes 15 hours there -- which is exactly what happened the
// first time this was run. Size the run before starting it; pass `limit` to
// trade power for turnaround.
#include "agents/search_agent.hpp"
#include "core/board.hpp"
#include "core/random.hpp"
#include "core/spawn.hpp"
#include "evaluation/n1_evaluator.hpp"
#include "game/game.hpp"
#include "learning/ntuple_network.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace a2048 = adversarial_2048;

namespace {

constexpr char kMagic[8] = {'A', '2', '0', '4', '8', 'J', 'C', 'T'};

struct Juncture {
    a2048::Board board;
    std::uint64_t rng_seed;
};

void save_junctures(const std::string& path, const std::vector<Juncture>& junctures) {
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    out.write(kMagic, sizeof(kMagic));
    const auto count = static_cast<std::uint64_t>(junctures.size());
    out.write(reinterpret_cast<const char*>(&count), sizeof(count));
    for (const auto& juncture : junctures) {
        out.write(reinterpret_cast<const char*>(&juncture.board.packed_exponents), 8);
        out.write(reinterpret_cast<const char*>(&juncture.board.exponent_high_bits), 2);
        out.write(reinterpret_cast<const char*>(&juncture.rng_seed), 8);
    }
}

std::vector<Juncture> load_junctures(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    std::array<char, sizeof(kMagic)> magic{};
    in.read(magic.data(), magic.size());
    if (!in || std::memcmp(magic.data(), kMagic, sizeof(kMagic)) != 0) {
        std::fprintf(stderr, "not a juncture file: %s\n", path.c_str());
        std::exit(2);
    }
    std::uint64_t count = 0;
    in.read(reinterpret_cast<char*>(&count), sizeof(count));
    std::vector<Juncture> junctures(count);
    for (auto& juncture : junctures) {
        in.read(reinterpret_cast<char*>(&juncture.board.packed_exponents), 8);
        in.read(reinterpret_cast<char*>(&juncture.board.exponent_high_bits), 2);
        in.read(reinterpret_cast<char*>(&juncture.rng_seed), 8);
    }
    return junctures;
}

// Largest and second-largest exponent on the board. The second has to reach 14
// for a 32768 to become possible, which is what makes it the quantity to track.
std::pair<std::uint8_t, std::uint8_t> top_two(a2048::Board board) {
    const auto cells = a2048::decode(board);
    std::uint8_t best = 0;
    std::uint8_t second = 0;
    for (const auto exponent : cells) {
        if (exponent > best) {
            second = best;
            best = exponent;
        } else if (exponent > second) {
            second = exponent;
        }
    }
    return {best, second};
}

// The board's condition AT THE JUNCTURE, before any of the tail is played, so
// the outcome can be related to how the agent arrived rather than only
// aggregated away. E34 found these do not predict conversion -- recorded so the
// next person does not have to rediscover that.
struct JunctureFeatures {
    std::uint8_t free_cells{};
    std::uint8_t snake_run{};
    std::uint8_t cornered{};
    std::uint8_t distinct_tiles{};
    std::uint32_t tile_sum{};
};

JunctureFeatures features_of(a2048::Board board) {
    const auto cells = a2048::decode(board);
    JunctureFeatures features{};
    features.free_cells = static_cast<std::uint8_t>(a2048::empty_count(board));

    // NOTE: this counts from cell 0 only, so it reads ~1 on almost every board
    // and is a weak measure. Kept as recorded, not as recommended.
    static constexpr std::array<std::size_t, 16> kSnake{
        0, 1, 2, 3, 7, 6, 5, 4, 8, 9, 10, 11, 15, 14, 13, 12};
    std::uint8_t previous = 16;
    for (const auto cell : kSnake) {
        const auto exponent = cells[cell];
        if (exponent == 0 || exponent > previous) {
            break;
        }
        previous = exponent;
        ++features.snake_run;
    }

    std::uint8_t best = 0;
    for (const auto exponent : cells) {
        best = std::max(best, exponent);
    }
    features.cornered = (cells[0] == best || cells[3] == best || cells[12] == best ||
                         cells[15] == best) ? 1U : 0U;

    std::array<bool, 16> seen{};
    for (const auto exponent : cells) {
        if (exponent != 0 && !seen[exponent]) {
            seen[exponent] = true;
            ++features.distinct_tiles;
        }
        features.tile_sum += exponent == 0 ? 0U : (1U << exponent);
    }
    return features;
}

a2048::ExpectimaxOptions make_options(double cutoff) {
    a2048::ExpectimaxOptions options;
    options.minimum_path_probability = cutoff;
    return options;
}

std::unique_ptr<a2048::SearchAgent> make_agent(
    const a2048::Evaluator& evaluator, const std::string& depth_spec, double cutoff) {
    if (depth_spec.find(',') != std::string::npos) {
        int high = 0;
        int medium = 0;
        int low = 0;
        std::sscanf(depth_spec.c_str(), "%d,%d,%d", &high, &medium, &low);
        return std::make_unique<a2048::SearchAgent>(
            evaluator,
            a2048::AdaptiveDepthSchedule{static_cast<std::uint32_t>(high),
                                         static_cast<std::uint32_t>(medium),
                                         static_cast<std::uint32_t>(low)},
            "expectimax", make_options(cutoff));
    }
    return std::make_unique<a2048::SearchAgent>(
        evaluator, static_cast<std::uint32_t>(std::atoi(depth_spec.c_str())),
        "expectimax", make_options(cutoff));
}

int collect(const a2048::Evaluator& evaluator, char** argv) {
    const std::string out = argv[3];
    const auto games = static_cast<std::size_t>(std::atoi(argv[4]));
    const auto first_seed = static_cast<std::uint64_t>(std::atoll(argv[5]));
    const auto threads = static_cast<std::size_t>(std::atoi(argv[6]));

    std::vector<Juncture> junctures;
    std::mutex mutex;
    std::atomic<std::size_t> next{0};
    std::atomic<std::size_t> played{0};
    const auto worker = [&] {
        a2048::SearchAgent agent(evaluator, 4, "expectimax", make_options(0.0015));
        while (true) {
            const auto index = next.fetch_add(1);
            if (index >= games) {
                return;
            }
            // Progress, and a periodic checkpoint. The first long collection ran
            // 12 hours writing nothing until the end, so there was no way to
            // tell progress from a hang and a kill would have forfeited all of
            // it. Cost scales with GAMES PLAYED, not junctures found: every game
            // runs to completion whether or not it reaches the juncture, at
            // ~93 core-seconds each, and 8 workers on this M1 is a measured
            // 3.43x -- not 8x. Size accordingly.
            const auto done = played.fetch_add(1) + 1;
            if (done % 50 == 0) {
                const std::lock_guard<std::mutex> guard(mutex);
                save_junctures(out, junctures);
                std::fprintf(stderr, "  %zu/%zu games, %zu junctures (checkpointed)\n",
                             done, games, junctures.size());
                std::fflush(stderr);
            }
            a2048::Game game(first_seed + index);
            bool captured = false;
            while (!game.game_over()) {
                const auto direction = agent.choose_move(game.board());
                if (!direction.has_value()) {
                    break;
                }
                static_cast<void>(game.apply_move(*direction));
                if (captured) {
                    continue;
                }
                const auto [best, second] = top_two(game.board());
                // Captured once, the first time it occurs, so every game
                // contributes at most one sample and long games are not
                // over-represented.
                if (best == 14 && second == 13) {
                    const std::lock_guard<std::mutex> guard(mutex);
                    junctures.push_back(Juncture{game.board(), first_seed + index});
                    captured = true;
                }
            }
        }
    };
    std::vector<std::thread> pool;
    for (std::size_t index = 1; index < threads; ++index) {
        pool.emplace_back(worker);
    }
    worker();
    for (auto& thread : pool) {
        thread.join();
    }
    std::sort(junctures.begin(), junctures.end(),
              [](const Juncture& left, const Juncture& right) {
                  return left.rng_seed < right.rng_seed;
              });
    save_junctures(out, junctures);
    std::printf("collected %zu junctures from %zu games (%.1f%%) -> %s\n",
                junctures.size(), games,
                100.0 * static_cast<double>(junctures.size()) / static_cast<double>(games),
                out.c_str());
    return 0;
}

int probe(const a2048::Evaluator& evaluator, int argc, char** argv) {
    const auto junctures = load_junctures(argv[3]);
    const std::string depth_spec = argv[4];
    const double cutoff = std::atof(argv[5]);
    const auto threads = static_cast<std::size_t>(std::atoi(argv[6]));
    const auto limit = argc > 7 ? static_cast<std::size_t>(std::atoi(argv[7]))
                                : junctures.size();
    const auto count = std::min(limit, junctures.size());

    std::FILE* dump = nullptr;
    if (const char* path = std::getenv("CONVERT_PROBE_DUMP")) {
        dump = std::fopen(path, "w");
        if (dump != nullptr) {
            std::fprintf(dump, "index,free_cells,snake_run,cornered,distinct_tiles,"
                               "tile_sum,converted,tail_moves\n");
        }
    }

    std::atomic<std::size_t> next{0};
    std::atomic<std::size_t> converted{0};
    std::atomic<std::size_t> moves_total{0};
    std::atomic<std::size_t> finished{0};
    std::mutex report_mutex;

    const auto worker = [&] {
        auto agent = make_agent(evaluator, depth_spec, cutoff);
        while (true) {
            const auto index = next.fetch_add(1);
            if (index >= count) {
                return;
            }
            // Continue from the juncture with a stream derived from its own
            // seed, so every arm replays the identical spawn sequence given the
            // same moves.
            a2048::Game game(junctures[index].board,
                             junctures[index].rng_seed ^ 0x5DEECE66DULL);
            std::size_t moves = 0;
            bool made_second = false;
            while (!game.game_over()) {
                const auto direction = agent->choose_move(game.board());
                if (!direction.has_value()) {
                    break;
                }
                static_cast<void>(game.apply_move(*direction));
                ++moves;
                const auto [best, second] = top_two(game.board());
                if (best >= 15 || (best == 14 && second == 14)) {
                    made_second = true;
                    break;
                }
            }
            moves_total += moves;
            if (made_second) {
                ++converted;
            }
            if (dump != nullptr) {
                const auto features = features_of(junctures[index].board);
                const std::lock_guard<std::mutex> guard(report_mutex);
                std::fprintf(dump, "%zu,%u,%u,%u,%u,%u,%d,%zu\n", index,
                             features.free_cells, features.snake_run, features.cornered,
                             features.distinct_tiles, features.tile_sum,
                             made_second ? 1 : 0, moves);
            }
            // Report as trials land. The first run of this tool took 15 hours
            // and printed nothing until it was done, so there was no way to
            // tell progress from a hang.
            const auto done = ++finished;
            if (done % 25 == 0 || done == count) {
                const std::lock_guard<std::mutex> guard(report_mutex);
                const auto hits = converted.load();
                std::fprintf(stderr, "  %zu/%zu trials, %zu converted (%.1f%%)\n", done,
                             count, hits,
                             100.0 * static_cast<double>(hits) / static_cast<double>(done));
                std::fflush(stderr);
            }
        }
    };
    std::vector<std::thread> pool;
    for (std::size_t index = 1; index < threads; ++index) {
        pool.emplace_back(worker);
    }
    worker();
    for (auto& thread : pool) {
        thread.join();
    }
    if (dump != nullptr) {
        std::fclose(dump);
    }

    const auto hits = converted.load();
    const auto rate = static_cast<double>(hits) / static_cast<double>(count);
    const auto half = 1.96 * std::sqrt(rate * (1.0 - rate) / static_cast<double>(count));
    std::printf("depth=%s cutoff=%g  n=%zu  converted %zu = %.2f%%  95%% CI [%.2f%%, %.2f%%]"
                "  mean tail moves %.0f\n",
                depth_spec.c_str(), cutoff, count, hits, 100.0 * rate,
                100.0 * std::max(0.0, rate - half), 100.0 * (rate + half),
                static_cast<double>(moves_total.load()) / static_cast<double>(count));
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 4) {
        std::fprintf(stderr,
                     "usage:\n"
                     "  convert_probe collect <weights> <out.bin> <games> <seed> <threads>\n"
                     "  convert_probe probe   <weights> <in.bin> <depth|H,M,L> <cutoff> "
                     "<threads> [limit]\n");
        return 2;
    }
    const std::string mode = argv[1];
    auto network = a2048::learning::NTupleNetwork::load_from(argv[2]);
    const a2048::N1Evaluator evaluator(network);
    if (mode == "collect") {
        return argc >= 7 ? collect(evaluator, argv) : 2;
    }
    if (mode == "probe") {
        return argc >= 7 ? probe(evaluator, argc, argv) : 2;
    }
    std::fprintf(stderr, "unknown mode: %s\n", mode.c_str());
    return 2;
}
