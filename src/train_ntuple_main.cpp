// Trains an n-tuple network by afterstate TD learning and writes the weights
// to a file that `run_experiment --heuristic N1 --weights <path>` can load.
//
// Training is 1-ply greedy self-play (no expectimax), matching the reference
// papers. That also sidesteps a real hazard: the search's transposition table
// keys on the board alone, so it would serve stale values if weights changed
// mid-search. Weights are frozen before any search-based evaluation.
#include "learning/ntuple_network.hpp"
#include "learning/td_trainer.hpp"

#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

namespace nn = adversarial_2048::learning;

namespace {

void print_usage(const char* program) {
    std::cerr
        << "Usage: " << program << " --out <path> [options]\n"
        << "  --out PATH            where to write weights (required)\n"
        << "  --games N             training games (default 100000)\n"
        << "  --alpha X             learning rate, per-weight step is alpha/m (default 0.1)\n"
        << "  --seed N              RNG seed (default 1)\n"
        << "  --resume PATH         load starting weights from PATH\n"
        << "  --checkpoint-every N  save weights every N games as <out>.atN (0 = off)\n"
        << "  --evaluate-every N    greedy evaluation every N games (default 10000, 0 = off)\n"
        << "  --evaluation-games N  games per evaluation (default 100)\n"
        << "  --temporal-coherence  per-weight adaptive step sizes (~3x memory)\n"
        << "  --optimistic-init X   start weights so an unseen board scores ~X\n"
        << "  --backward-updates    replay each episode in reverse (faster terminal signal)\n"
        << "  --tuples NAME         shape: default 128MB | large 320MB | xlarge 512MB\n"
        << "  --stages N            independent weight sets by max tile (default 1)\n"
        << "  --global-features     add a whole-board (empties x max tile) feature\n"
        << "  --tc-state PATH       persist TC accumulators here (enables real resume)\n"
        << "  --train-depth N       pick training actions by depth-N search (default 1)\n";
}

[[nodiscard]] std::uint64_t parse_u64(const std::string& text, const std::string& flag) {
    try {
        std::size_t consumed = 0;
        const auto value = std::stoull(text, &consumed);
        if (consumed != text.size()) {
            throw std::invalid_argument("");
        }
        return value;
    } catch (const std::exception&) {
        throw std::invalid_argument("invalid " + flag + ": '" + text + "'");
    }
}

[[nodiscard]] double parse_double(const std::string& text, const std::string& flag) {
    try {
        std::size_t consumed = 0;
        const auto value = std::stod(text, &consumed);
        if (consumed != text.size()) {
            throw std::invalid_argument("");
        }
        return value;
    } catch (const std::exception&) {
        throw std::invalid_argument("invalid " + flag + ": '" + text + "'");
    }
}

[[nodiscard]] const std::string& require_value(
    const std::vector<std::string>& args, std::size_t& index, const std::string& flag) {
    if (index + 1 >= args.size()) {
        throw std::invalid_argument(flag + " requires a value");
    }
    return args[++index];
}

}  // namespace

int main(int argc, char* argv[]) {
    const std::vector<std::string> args(argv + 1, argv + argc);

    std::filesystem::path out;
    std::filesystem::path resume;
    std::string tuples = "default";
    std::uint64_t stages = 1;
    bool global_features = false;
    nn::TrainConfig config;
    config.checkpoint_every = 10000;
    config.evaluate_every = 10000;

    try {
        for (std::size_t index = 0; index < args.size(); ++index) {
            const auto& flag = args[index];
            if (flag == "--out") {
                out = require_value(args, index, flag);
            } else if (flag == "--resume") {
                resume = require_value(args, index, flag);
            } else if (flag == "--games") {
                config.games = parse_u64(require_value(args, index, flag), flag);
            } else if (flag == "--alpha") {
                config.alpha = parse_double(require_value(args, index, flag), flag);
            } else if (flag == "--seed") {
                config.seed = parse_u64(require_value(args, index, flag), flag);
            } else if (flag == "--checkpoint-every") {
                config.checkpoint_every = parse_u64(require_value(args, index, flag), flag);
            } else if (flag == "--evaluate-every") {
                config.evaluate_every = parse_u64(require_value(args, index, flag), flag);
            } else if (flag == "--evaluation-games") {
                config.evaluation_games = parse_u64(require_value(args, index, flag), flag);
            } else if (flag == "--train-depth") {
                config.training_search_depth =
                    static_cast<std::uint32_t>(parse_u64(require_value(args, index, flag), flag));
            } else if (flag == "--tc-state") {
                config.temporal_coherence_state = require_value(args, index, flag);
            } else if (flag == "--global-features") {
                global_features = true;
            } else if (flag == "--stages") {
                stages = parse_u64(require_value(args, index, flag), flag);
            } else if (flag == "--tuples") {
                tuples = require_value(args, index, flag);
            } else if (flag == "--backward-updates") {
                config.backward_updates = true;
            } else if (flag == "--temporal-coherence") {
                config.temporal_coherence = true;
            } else if (flag == "--optimistic-init") {
                config.optimistic_initial_value =
                    std::stod(require_value(args, index, flag));
            } else {
                throw std::invalid_argument("unrecognized argument: '" + flag + "'");
            }
        }
        if (out.empty()) {
            throw std::invalid_argument("--out is required");
        }
        if (config.alpha <= 0.0) {
            throw std::invalid_argument("--alpha must be positive");
        }
        // Optimistic initialisation overwrites every weight, so combining it
        // with --resume would silently discard the network being resumed.
        if (!resume.empty() && config.optimistic_initial_value != 0.0) {
            throw std::invalid_argument(
                "--optimistic-init overwrites all weights and cannot be combined with --resume");
        }
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << '\n';
        print_usage(argv[0]);
        return 2;
    }

    try {
        nn::NTupleNetwork network(nn::named_tuple_specs(tuples),
                                  static_cast<std::size_t>(stages), global_features);
        if (!resume.empty()) {
            network.load(resume);
            std::cout << "resumed from " << resume << '\n';
            // Only meaningful when a state file was supplied AND already
            // exists; a first run with --tc-state should start fresh.
            if (!config.temporal_coherence_state.empty() &&
                std::filesystem::exists(config.temporal_coherence_state)) {
                config.resume_temporal_coherence = true;
                std::cout << "resuming TC accumulators from "
                          << config.temporal_coherence_state << '\n';
            } else if (config.temporal_coherence) {
                std::cout << "WARNING: resuming weights with FRESH temporal-coherence state;\n"
                          << "         every beta restarts at 1.0, so early steps will be "
                          << "large.\n";
            }
        }

        std::cout << "network: " << network.specs().size() << " tuples, "
                  << network.active_weight_count() << " active weights/eval, "
                  << network.total_weight_count() << " total ("
                  << (network.total_weight_count() * sizeof(float)) / (1024 * 1024)
                  << " MB)\n"
                  << "training " << config.games << " games, alpha " << config.alpha
                  << ", seed " << config.seed
                  << (config.temporal_coherence ? ", temporal coherence" : "")
                  << '\n';
        if (config.training_search_depth > 1) {
            std::cout << "  training actions chosen by depth-" << config.training_search_depth
                      << " search (transposition table off)\n";
        }
        if (network.has_global_features()) {
            std::cout << "  global features: on (whole-board empties x max tile)\n";
        }
        if (network.stage_count() > 1) {
            std::cout << "  stages: " << network.stage_count()
                      << " independent weight sets (by max tile)\n";
        }
        if (config.temporal_coherence) {
            std::cout << "  TC state: +"
                      << (2 * network.total_weight_count() * sizeof(float)) / (1024 * 1024)
                      << " MB"
                      << (config.temporal_coherence_state.empty()
                              ? " (not persisted; this run cannot be resumed)"
                              : " -> " + config.temporal_coherence_state.string())
                      << '\n';
        }
        if (config.optimistic_initial_value != 0.0) {
            std::cout << "  optimistic init: unseen board evaluates to ~"
                      << config.optimistic_initial_value << '\n';
        }
        std::cout.flush();

        const auto start = std::chrono::steady_clock::now();
        const auto result = nn::train(network, config, [&](std::uint64_t games) {
            // Checkpoints are VERSIONED by game count rather than overwriting
            // one file. Two reasons: an interrupted run never destroys its own
            // last good snapshot, and the intermediate networks are research
            // data in their own right — they are the only way to ask how a
            // property (say, whether search depth still pays) changes with
            // training budget, which cannot be reconstructed after the fact.
            auto checkpoint = out;
            checkpoint.replace_extension();
            checkpoint += ".at" + std::to_string(games) + out.extension().string();
            network.save(checkpoint);
            const auto elapsed =
                std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
            std::cout << "  checkpoint at " << games << " games (" << std::fixed
                      << std::setprecision(1) << elapsed << " s) -> "
                      << checkpoint.filename().string() << '\n';
            std::cout.flush();
        });
        const auto elapsed =
            std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();

        network.save(out);

        std::cout << std::fixed << std::setprecision(1)
                  << "done in " << elapsed << " s\n"
                  << "training mean score " << result.mean_score
                  << ", best " << result.best_score
                  << ", highest tile " << result.highest_tile << '\n';
        if (config.temporal_coherence) {
            // Starts at 1.0 and falls as weights stop oscillating. A value
            // still near 1.0 means TC never damped anything and the run was
            // effectively plain TD.
            std::cout << "mean temporal-coherence beta " << std::setprecision(4)
                      << result.final_mean_beta << std::setprecision(1) << '\n';
        }
        if (!result.learning_curve.empty()) {
            std::cout << "learning curve (greedy 1-ply, held-out seeds):\n";
            for (const auto& sample : result.learning_curve) {
                std::cout << "  " << std::setw(10) << sample.games_played << " games: "
                          << std::setw(12) << std::setprecision(1) << sample.mean_score
                          << "  max tile " << sample.max_tile << '\n';
            }
        }
        std::cout << "wrote " << out << "\n  " << network.fingerprint() << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << '\n';
        return 1;
    }
}
