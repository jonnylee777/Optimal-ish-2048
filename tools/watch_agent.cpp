// Watch an agent play, live, in the terminal.
//
// Every other entry point in this project reduces a game to a row in a CSV.
// That is the right thing for measurement and the wrong thing for understanding
// what the agent is actually doing -- the 32768 investigation
// (docs/32768-investigation.md) turned on noticing *how* games were being lost,
// which no aggregate could have shown.
//
// Any trained network or hand-written evaluator can be loaded, so versions can
// be watched side by side in two terminals, or in sequence.
//
//   watch_agent --weights experiments/weights/n25_best_2M5.bin
//   watch_agent --heuristic H5 --depth 4 --delay-ms 40
//   watch_agent --weights <path> --from-tile 8192   # sprint, then slow down
//   watch_agent --list                              # available networks
//
// The interesting part of a game is its last few hundred moves; the first
// several thousand are a formality. `--from-tile` plays at full speed until the
// board reaches that tile and only then slows to the watchable rate, which is
// what makes a 15,000-move game worth sitting through.
#include "agents/search_agent.hpp"
#include "core/board.hpp"
#include "evaluation/h0_heuristic.hpp"
#include "evaluation/h2_heuristic.hpp"
#include "evaluation/h3_heuristic.hpp"
#include "evaluation/h4_heuristic.hpp"
#include "evaluation/h5_heuristic.hpp"
#include "evaluation/n1_evaluator.hpp"
#include "evaluation/baseline_heuristic.hpp"
#include "evaluation/structural_heuristic.hpp"
#include "game/game.hpp"
#include "learning/ntuple_network.hpp"

#include <algorithm>
#include <chrono>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <memory>
#include <string>
#include <thread>
#include <vector>

namespace a2048 = adversarial_2048;

namespace {

// ---------------------------------------------------------------- terminal --

constexpr const char* kHome = "\033[H";
constexpr const char* kClear = "\033[2J";
constexpr const char* kHideCursor = "\033[?25l";
constexpr const char* kShowCursor = "\033[?25h";
constexpr const char* kReset = "\033[0m";
constexpr const char* kDim = "\033[38;2;130;128;120m";
constexpr const char* kBold = "\033[1m";

void restore_terminal() {
    // Registered with atexit AND called on the normal path, so it has to be
    // safe to run twice.
    static bool restored = false;
    if (restored) return;
    restored = true;
    std::fputs(kShowCursor, stdout);
    std::fflush(stdout);
}

// The cursor is hidden for the duration of the run, so it has to be restored on
// every exit path -- including Ctrl-C, which would otherwise leave the user's
// shell with no cursor.
extern "C" void on_signal(int sig) {
    restore_terminal();
    std::signal(sig, SIG_DFL);
    std::raise(sig);
}

struct Rgb { int r, g, b; };

// The familiar 2048 colours. Tiles also carry their number, so colour is never
// the only thing distinguishing them.
Rgb tile_background(std::uint8_t exponent) {
    switch (exponent) {
        case 0:  return {205, 193, 180};
        case 1:  return {238, 228, 218};
        case 2:  return {237, 224, 200};
        case 3:  return {242, 177, 121};
        case 4:  return {245, 149,  99};
        case 5:  return {246, 124,  95};
        case 6:  return {246,  94,  59};
        case 7:  return {237, 207, 114};
        case 8:  return {237, 204,  97};
        case 9:  return {237, 200,  80};
        case 10: return {237, 197,  63};
        case 11: return {237, 194,  46};
        case 12: return { 60,  58,  50};
        default: return { 60,  58,  50};
    }
}

Rgb tile_foreground(std::uint8_t exponent) {
    return (exponent >= 1 && exponent <= 2) ? Rgb{119, 110, 101} : Rgb{249, 246, 242};
}

void put_colour(const Rgb& fg, const Rgb& bg) {
    std::printf("\033[38;2;%d;%d;%dm\033[48;2;%d;%d;%dm", fg.r, fg.g, fg.b, bg.r, bg.g, bg.b);
}

std::string tile_text(std::uint8_t exponent) {
    if (exponent == 0) return "";
    return std::to_string(1ULL << exponent);
}

void draw_board(a2048::Board board) {
    const auto cells = a2048::decode(board);
    constexpr int kWidth = 8;
    for (std::size_t row = 0; row < a2048::kBoardWidth; ++row) {
        // Three terminal lines per board row gives roughly square tiles.
        for (int line = 0; line < 3; ++line) {
            std::fputs("  ", stdout);
            for (std::size_t col = 0; col < a2048::kBoardWidth; ++col) {
                const auto exponent = cells[row * a2048::kBoardWidth + col];
                put_colour(tile_foreground(exponent), tile_background(exponent));
                if (line == 1) {
                    const auto text = tile_text(exponent);
                    const int pad = kWidth - static_cast<int>(text.size());
                    const int left = pad / 2;
                    std::printf("%*s%s%*s", left, "", text.c_str(), pad - left, "");
                } else {
                    std::printf("%*s", kWidth, "");
                }
                std::fputs(kReset, stdout);
            }
            std::fputs("\n", stdout);
        }
    }
}

const char* direction_name(a2048::Direction d) {
    switch (d) {
        case a2048::Direction::up:    return "up";
        case a2048::Direction::down:  return "down";
        case a2048::Direction::left:  return "left";
        case a2048::Direction::right: return "right";
    }
    return "?";
}

// ------------------------------------------------------------------ config --

struct Options {
    std::string heuristic = "N1";
    std::filesystem::path weights = "experiments/weights/n25_best_2M5.bin";
    std::uint32_t depth = 4;
    double cutoff = 0.0015;
    std::uint64_t seed = 30000;
    std::size_t games = 1;
    int delay_ms = 60;
    bool step = false;
    std::uint64_t from_tile = 0;
    bool list = false;
    // Dump every board to a file instead of (also) drawing it, so a game can be
    // turned into an animation offline. Recording implies no delay and no
    // drawing: a 22,000-move game is not worth watching in real time, and the
    // point is to capture it exactly, once.
    std::filesystem::path record;
};

[[noreturn]] void usage(int code) {
    std::fputs(
        "Watch a 2048 agent play, live.\n\n"
        "  --weights PATH        trained network to watch (default: best agent)\n"
        "  --heuristic NAME      N1 (learned) or H0..H5 (hand-written). Default N1\n"
        "  --depth N             moves searched ahead (default 4)\n"
        "  --probability-cutoff X  search pruning threshold (default 0.0015)\n"
        "  --seed N              which game to play (default 30000)\n"
        "  --games N             play N games in a row (default 1)\n"
        "  --delay-ms N          pause between moves; 0 for as fast as it draws\n"
        "  --step                advance one move per Enter keypress\n"
        "  --from-tile N         play at full speed until this tile appears\n"
        "  --list                list available trained networks\n"
        "  --record PATH         write every board to PATH instead of drawing\n",
        code == 0 ? stdout : stderr);
    std::exit(code);
}

std::string need(int argc, char** argv, int& i, const char* flag) {
    if (i + 1 >= argc) {
        std::fprintf(stderr, "error: %s needs a value\n", flag);
        std::exit(2);
    }
    return argv[++i];
}

Options parse(int argc, char** argv) {
    Options o;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--weights")                  o.weights = need(argc, argv, i, "--weights");
        else if (a == "--heuristic")           o.heuristic = need(argc, argv, i, "--heuristic");
        else if (a == "--depth")               o.depth = static_cast<std::uint32_t>(std::stoul(need(argc, argv, i, "--depth")));
        else if (a == "--probability-cutoff")  o.cutoff = std::stod(need(argc, argv, i, "--probability-cutoff"));
        else if (a == "--seed")                o.seed = std::stoull(need(argc, argv, i, "--seed"));
        else if (a == "--games")               o.games = std::stoul(need(argc, argv, i, "--games"));
        else if (a == "--delay-ms")            o.delay_ms = std::stoi(need(argc, argv, i, "--delay-ms"));
        else if (a == "--step")                o.step = true;
        else if (a == "--from-tile")           o.from_tile = std::stoull(need(argc, argv, i, "--from-tile"));
        else if (a == "--list")                o.list = true;
        else if (a == "--record")              o.record = need(argc, argv, i, "--record");
        else if (a == "--help" || a == "-h")   usage(0);
        else { std::fprintf(stderr, "error: unrecognized argument '%s'\n", a.c_str()); usage(2); }
    }
    if (o.depth == 0) { std::fputs("error: --depth must be at least 1\n", stderr); std::exit(2); }
    return o;
}

void list_networks() {
    const std::filesystem::path dir = "experiments/weights";
    if (!std::filesystem::exists(dir)) {
        std::fprintf(stderr, "no %s directory here -- run from the repository root\n",
                     dir.c_str());
        std::exit(2);
    }
    std::vector<std::pair<std::string, std::uintmax_t>> found;
    for (const auto& entry : std::filesystem::directory_iterator(dir)) {
        if (entry.path().extension() == ".bin" &&
            entry.path().filename().string().rfind("n", 0) == 0) {
            found.emplace_back(entry.path().filename().string(), entry.file_size());
        }
    }
    std::sort(found.begin(), found.end());
    std::printf("Trained networks in %s:\n\n", dir.c_str());
    for (const auto& [name, size] : found) {
        std::printf("  %-34s %6ju MB\n", name.c_str(),
                    static_cast<std::uintmax_t>(size / (1024 * 1024)));
    }
    std::printf("\nWatch one with:  watch_agent --weights %s/<name>\n", dir.c_str());
}

}  // namespace

int main(int argc, char** argv) {
    const auto options = parse(argc, argv);
    if (options.list) { list_networks(); return 0; }

    // Hand-written evaluators are constructed with their tuned defaults; only
    // the learned network needs a file. Held by unique_ptr because the search
    // borrows the evaluator by reference for the whole run.
    std::unique_ptr<a2048::learning::NTupleNetwork> network;
    std::unique_ptr<a2048::Evaluator> evaluator;
    std::string label = options.heuristic;
    try {
        if (options.heuristic == "N1") {
            if (!std::filesystem::exists(options.weights)) {
                std::fprintf(stderr, "error: no weight file at %s\n"
                                     "       try --list, or pass --heuristic H5 for a "
                                     "hand-written evaluator\n",
                             options.weights.c_str());
                return 2;
            }
            network = std::make_unique<a2048::learning::NTupleNetwork>(
                a2048::learning::NTupleNetwork::load_from(options.weights));
            evaluator = std::make_unique<a2048::N1Evaluator>(*network);
            label = "learned network " + options.weights.filename().string();
        } else if (options.heuristic == "H0") {
            evaluator = std::make_unique<a2048::H0Heuristic>();
        } else if (options.heuristic == "H1") {
            evaluator = std::make_unique<a2048::BaselineHeuristic>();
        } else if (options.heuristic == "H2") {
            evaluator = std::make_unique<a2048::H2Heuristic>();
        } else if (options.heuristic == "H3") {
            evaluator = std::make_unique<a2048::StructuralHeuristic>();
        } else if (options.heuristic == "H4") {
            evaluator = std::make_unique<a2048::H4Heuristic>();
        } else if (options.heuristic == "H5") {
            evaluator = std::make_unique<a2048::H5Heuristic>();
        } else {
            std::fprintf(stderr, "error: unknown evaluator '%s' (N1, H0..H5)\n",
                         options.heuristic.c_str());
            return 2;
        }
    } catch (const std::exception& error) {
        std::fprintf(stderr, "error: %s\n", error.what());
        return 2;
    }

    // Recording writes one line per move: move number, score, and the packed
    // board. Kept as plain text rather than a binary dump so the file can be
    // read by the rendering script without a shared format definition.
    std::FILE* record = nullptr;
    if (!options.record.empty()) {
        record = std::fopen(options.record.c_str(), "w");
        if (record == nullptr) {
            std::fprintf(stderr, "error: cannot write %s\n", options.record.c_str());
            return 2;
        }
        std::fprintf(record, "move,score,packed,high\n");
    }

    std::signal(SIGINT, on_signal);
    std::signal(SIGTERM, on_signal);
    std::atexit(restore_terminal);

    a2048::ExpectimaxOptions search_options;
    search_options.minimum_path_probability = options.cutoff;
    a2048::SearchAgent agent(*evaluator, options.depth, "expectimax", search_options);

    std::fputs(kHideCursor, stdout);
    std::fputs(kClear, stdout);

    std::uint64_t best_score = 0;
    std::uint8_t best_tile = 0;
    for (std::size_t index = 0; index < options.games; ++index) {
        const auto seed = options.seed + index;
        a2048::Game game(seed);
        double move_ms = 0.0;
        bool sprinting = options.from_tile > 0;

        while (!game.game_over()) {
            const auto started = std::chrono::steady_clock::now();
            const auto direction = agent.choose_move(game.board());
            if (!direction.has_value()) break;
            move_ms = std::chrono::duration<double, std::milli>(
                          std::chrono::steady_clock::now() - started).count();
            static_cast<void>(game.apply_move(*direction));

            if (record != nullptr) {
                std::fprintf(record, "%llu,%llu,%llu,%u\n",
                             static_cast<unsigned long long>(game.move_count()),
                             static_cast<unsigned long long>(game.score()),
                             static_cast<unsigned long long>(game.board().packed_exponents),
                             static_cast<unsigned>(game.board().exponent_high_bits));
                continue;  // recording is a capture, not a viewing
            }

            const auto peak = a2048::max_exponent(game.board());
            if (sprinting && peak != 0 && (1ULL << peak) >= options.from_tile) {
                sprinting = false;
            }
            // While sprinting, redraw occasionally rather than never -- a frozen
            // screen is indistinguishable from a hang.
            const bool draw = !sprinting || game.move_count() % 200 == 0;
            if (!draw) continue;

            std::fputs(kHome, stdout);
            std::printf("  %s%s%s", kBold, label.c_str(), kReset);
            std::printf("%s   depth %u   game %zu of %zu (seed %llu)%s\n\n",
                        kDim, options.depth, index + 1, options.games,
                        static_cast<unsigned long long>(seed), kReset);
            draw_board(game.board());
            std::printf("\n  score %s%llu%s    move %llu    largest %llu\n",
                        kBold, static_cast<unsigned long long>(game.score()), kReset,
                        static_cast<unsigned long long>(game.move_count()),
                        peak == 0 ? 0ULL : (1ULL << peak));
            // Sub-millisecond moves are normal at shallow depth, so scale the
            // unit rather than rendering everything below 0.05 ms as "0.0".
            char timing[64];
            if (move_ms >= 1.0) {
                std::snprintf(timing, sizeof(timing), "%.1f ms/move", move_ms);
            } else {
                std::snprintf(timing, sizeof(timing), "%.0f us/move", move_ms * 1000.0);
            }
            std::printf("  %slast move %-5s   %-14s%s%s\n", kDim,
                        direction_name(*direction), timing,
                        sprinting ? "  (fast-forwarding)" : "", kReset);
            std::printf("  %sCtrl-C to stop%s          \n", kDim, kReset);
            std::fflush(stdout);

            if (sprinting) continue;
            if (options.step) {
                std::string ignored;
                std::getline(std::cin, ignored);
            } else if (options.delay_ms > 0) {
                std::this_thread::sleep_for(std::chrono::milliseconds(options.delay_ms));
            }
        }

        const auto peak = a2048::max_exponent(game.board());
        if (record != nullptr) {
            std::fclose(record);
            record = nullptr;
            std::printf("recorded %llu moves to %s\n",
                        static_cast<unsigned long long>(game.move_count()),
                        options.record.c_str());
        }
        best_score = std::max(best_score, game.score());
        best_tile = std::max(best_tile, peak);
        std::printf("\n  %sgame over%s  score %llu, largest tile %llu, %llu moves\n",
                    kBold, kReset, static_cast<unsigned long long>(game.score()),
                    peak == 0 ? 0ULL : (1ULL << peak),
                    static_cast<unsigned long long>(game.move_count()));
        if (options.games > 1 && index + 1 < options.games) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1200));
            std::fputs(kClear, stdout);
        }
    }
    if (options.games > 1) {
        std::printf("\n  %s%zu games: best score %llu, best tile %llu%s\n", kBold,
                    options.games, static_cast<unsigned long long>(best_score),
                    best_tile == 0 ? 0ULL : (1ULL << best_tile), kReset);
    }
    restore_terminal();
    return 0;
}
