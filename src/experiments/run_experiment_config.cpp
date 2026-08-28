#include "experiments/run_experiment_config.hpp"
#include "experiments/seed_sets.hpp"

#include <stdexcept>

namespace adversarial_2048 {
namespace {

[[nodiscard]] std::uint64_t parse_u64(const std::string& text, const std::string& what) {
    try {
        std::size_t consumed = 0;
        const auto value = std::stoull(text, &consumed);
        if (consumed != text.size()) {
            throw std::invalid_argument("");
        }
        return value;
    } catch (const std::exception&) {
        throw std::invalid_argument("invalid " + what + ": '" + text + "'");
    }
}

[[nodiscard]] double parse_f64(const std::string& text, const std::string& what) {
    try {
        std::size_t consumed = 0;
        const auto value = std::stod(text, &consumed);
        if (consumed != text.size()) {
            throw std::invalid_argument("");
        }
        return value;
    } catch (const std::exception&) {
        throw std::invalid_argument("invalid " + what + ": '" + text + "'");
    }
}

[[nodiscard]] bool parse_on_off(const std::string& text, const std::string& what) {
    if (text == "on") {
        return true;
    }
    if (text == "off") {
        return false;
    }
    throw std::invalid_argument("invalid " + what + ": expected 'on' or 'off', got '" + text + "'");
}

[[nodiscard]] std::vector<std::string> split(const std::string& text, char delimiter) {
    std::vector<std::string> parts;
    std::string current;
    for (const auto character : text) {
        if (character == delimiter) {
            parts.push_back(current);
            current.clear();
        } else {
            current += character;
        }
    }
    parts.push_back(current);
    return parts;
}

[[nodiscard]] const std::string& require_value(
    const std::vector<std::string>& args, std::size_t& index, const std::string& flag) {
    if (index + 1 >= args.size()) {
        throw std::invalid_argument(flag + " requires a value");
    }
    return args[++index];
}

}  // namespace

ResolvedSeeds resolve_seed_spec(
    const std::string& spec, std::optional<std::size_t> games_override) {
    if (spec.find('-') != std::string::npos && spec != "quick" && spec != "standard" &&
        spec != "final") {
        const auto parts = split(spec, '-');
        if (parts.size() != 2) {
            throw std::invalid_argument("invalid --seeds range: '" + spec + "'");
        }
        if (games_override.has_value()) {
            throw std::invalid_argument("--games cannot be combined with a custom --seeds range");
        }
        const auto first = parse_u64(parts[0], "--seeds range start");
        const auto last = parse_u64(parts[1], "--seeds range end");
        if (last < first) {
            throw std::invalid_argument("--seeds range end must not precede start");
        }
        const auto count = static_cast<std::size_t>(last - first + 1U);
        return {RunConfig{count, first}, "custom"};
    }

    RunConfig set;
    std::string label;
    if (spec == "quick") {
        set = seed_sets::quick_benchmark;
        label = "quick-benchmark";
    } else if (spec == "standard") {
        set = seed_sets::standard_benchmark;
        label = "standard-benchmark";
    } else if (spec == "final") {
        set = seed_sets::final_benchmark;
        label = "final-benchmark";
    } else {
        throw std::invalid_argument(
            "invalid --seeds value: '" + spec + "' (expected quick, standard, final, or FIRST-LAST)");
    }

    if (games_override.has_value()) {
        if (*games_override == 0 || *games_override > set.game_count) {
            throw std::invalid_argument(
                "--games must be between 1 and the named set's size (" +
                std::to_string(set.game_count) + ")");
        }
        set.game_count = *games_override;
    }
    return {set, label};
}

RunExperimentConfig parse_run_experiment_args(const std::vector<std::string>& args) {
    RunExperimentConfig config;
    bool heuristic_set = false;
    bool search_set = false;
    bool depth_given = false;
    bool time_limit_given = false;
    bool adaptive_schedule_given = false;
    std::optional<std::string> seeds_spec;
    std::optional<std::size_t> games_override;

    for (std::size_t index = 0; index < args.size(); ++index) {
        const auto& flag = args[index];
        if (flag == "--heuristic") {
            const auto& value = require_value(args, index, flag);
            if (value == "H0") {
                config.heuristic = HeuristicChoice::h0;
            } else if (value == "H1") {
                config.heuristic = HeuristicChoice::h1;
            } else if (value == "H2") {
                config.heuristic = HeuristicChoice::h2;
            } else if (value == "H3") {
                config.heuristic = HeuristicChoice::h3;
            } else if (value == "H4") {
                config.heuristic = HeuristicChoice::h4;
            } else if (value == "H5") {
                config.heuristic = HeuristicChoice::h5;
            } else if (value == "N1") {
                config.heuristic = HeuristicChoice::n1;
            } else {
                throw std::invalid_argument(
                    "invalid --heuristic: '" + value +
                    "' (expected H0, H1, H2, H3, H4, H5, or N1)");
            }
            heuristic_set = true;
        } else if (flag == "--weight") {
            const auto& value = require_value(args, index, flag);
            const auto equals = value.find('=');
            if (equals == std::string::npos) {
                throw std::invalid_argument("invalid --weight (expected name=value): '" + value + "'");
            }
            config.weight_overrides.push_back(NamedParameter{
                value.substr(0, equals),
                parse_f64(value.substr(equals + 1), "--weight value")});
        } else if (flag == "--search") {
            const auto& value = require_value(args, index, flag);
            if (value == "fixed") {
                config.search = SearchMode::fixed;
            } else if (value == "timed") {
                config.search = SearchMode::timed;
            } else {
                throw std::invalid_argument("invalid --search: '" + value + "' (expected fixed or timed)");
            }
            search_set = true;
        } else if (flag == "--depth") {
            config.fixed_depth = static_cast<std::uint32_t>(
                parse_u64(require_value(args, index, flag), "--depth"));
            depth_given = true;
        } else if (flag == "--time-limit-ms") {
            config.time_limit_ms = parse_f64(require_value(args, index, flag), "--time-limit-ms");
            time_limit_given = true;
        } else if (flag == "--adaptive-schedule") {
            const auto parts = split(require_value(args, index, flag), ',');
            if (parts.size() != 3) {
                throw std::invalid_argument("--adaptive-schedule expects three comma-separated depths, e.g. 4,6,8");
            }
            config.adaptive_depths = AdaptiveDepths{
                static_cast<std::uint32_t>(parse_u64(parts[0], "--adaptive-schedule")),
                static_cast<std::uint32_t>(parse_u64(parts[1], "--adaptive-schedule")),
                static_cast<std::uint32_t>(parse_u64(parts[2], "--adaptive-schedule")),
            };
            adaptive_schedule_given = true;
        } else if (flag == "--transposition-table") {
            config.transposition_table = parse_on_off(require_value(args, index, flag), flag);
        } else if (flag == "--tt-capacity") {
            config.transposition_table_capacity = static_cast<std::size_t>(
                parse_u64(require_value(args, index, flag), "--tt-capacity"));
        } else if (flag == "--threads") {
            config.worker_threads = static_cast<std::size_t>(
                parse_u64(require_value(args, index, flag), "--threads"));
            if (config.worker_threads == 0) {
                throw std::invalid_argument("--threads must be at least 1");
            }
        } else if (flag == "--probability-cutoff") {
            config.probability_cutoff = parse_f64(require_value(args, index, flag), "--probability-cutoff");
        } else if (flag == "--symmetry") {
            config.symmetry = parse_on_off(require_value(args, index, flag), flag);
        } else if (flag == "--seeds") {
            seeds_spec = require_value(args, index, flag);
        } else if (flag == "--games") {
            games_override = static_cast<std::size_t>(
                parse_u64(require_value(args, index, flag), "--games"));
        } else if (flag == "--weights") {
            config.weights_path = require_value(args, index, flag);
        } else if (flag == "--output-dir") {
            config.output_dir = require_value(args, index, flag);
        } else if (flag == "--quiet") {
            config.quiet = true;
        } else {
            throw std::invalid_argument("unrecognized argument: '" + flag + "'");
        }
    }

    if (!heuristic_set) {
        throw std::invalid_argument("--heuristic is required (H0-H5 or N1)");
    }
    if (!search_set) {
        throw std::invalid_argument("--search is required (fixed or timed)");
    }
    if (!seeds_spec.has_value()) {
        throw std::invalid_argument("--seeds is required (quick, standard, final, or FIRST-LAST)");
    }

    if (config.search == SearchMode::fixed) {
        // The depth SCHEDULE and the time LIMIT are independent choices, and
        // coupling them used to make the schedule untestable. "Search deeper
        // when few cells are empty" is a statement about where depth is worth
        // spending; it does not require a deadline, and pairing it with one
        // makes the result depend on machine speed and on how many games run
        // concurrently -- so it could be neither reproduced nor parallelised.
        //
        // Fixed + schedule means: pick the depth by empty count, always
        // complete it. Deterministic, so --threads applies and scores are
        // identical at any worker count.
        //
        // This matters because a 160-game autopsy found 83.6% of games reach
        // 16384 + 8192 -- one merge short of a second 16384 -- and fail to
        // convert. That is a tight-board execution problem, exactly where a
        // low-empty-cell depth bump is aimed, and the schedule had never once
        // been run with a learned evaluator.
        if (!depth_given && !adaptive_schedule_given) {
            throw std::invalid_argument(
                "--depth or --adaptive-schedule is required for --search fixed");
        }
        if (depth_given && adaptive_schedule_given) {
            throw std::invalid_argument("--depth and --adaptive-schedule are mutually exclusive");
        }
        if (depth_given && config.fixed_depth == 0) {
            throw std::invalid_argument("--depth must be at least 1");
        }
        if (adaptive_schedule_given &&
            (config.adaptive_depths.high_empty_depth == 0 ||
             config.adaptive_depths.medium_empty_depth == 0 ||
             config.adaptive_depths.low_empty_depth == 0)) {
            throw std::invalid_argument("--adaptive-schedule depths must be at least 1");
        }
        if (time_limit_given) {
            throw std::invalid_argument("--time-limit-ms is not valid with --search fixed");
        }
        config.time_limit_ms = 0.0;
        config.use_adaptive_schedule = adaptive_schedule_given;
    } else {
        if (!time_limit_given || config.time_limit_ms <= 0.0) {
            throw std::invalid_argument("--time-limit-ms (> 0) is required for --search timed");
        }
        if (depth_given && adaptive_schedule_given) {
            throw std::invalid_argument("--depth and --adaptive-schedule are mutually exclusive for --search timed");
        }
        if (depth_given) {
            if (config.fixed_depth == 0) {
                throw std::invalid_argument("--depth must be at least 1");
            }
            config.use_adaptive_schedule = false;
        } else {
            config.use_adaptive_schedule = true;
            if (config.adaptive_depths.high_empty_depth == 0 ||
                config.adaptive_depths.medium_empty_depth == 0 ||
                config.adaptive_depths.low_empty_depth == 0) {
                throw std::invalid_argument("--adaptive-schedule depths must be at least 1");
            }
        }
    }

    if (config.heuristic == HeuristicChoice::n1) {
        if (config.weights_path.empty()) {
            throw std::invalid_argument(
                "--weights is required for N1 (train one with the train_ntuple binary)");
        }
    } else if (!config.weights_path.empty()) {
        throw std::invalid_argument("--weights only applies to learned evaluators (N-series)");
    }

    const auto resolved = resolve_seed_spec(*seeds_spec, games_override);
    config.run_config = resolved.run_config;
    config.seed_set_label = resolved.label;

    if (config.output_dir.empty()) {
        // Results are filed by METHODOLOGY, not by search regime: the H-series
        // (hand-written heuristics) and the N-series (learned networks) are
        // different experiments that happen to share this harness, and mixing
        // them in one directory invites comparing across seed sets. Both
        // regimes for one methodology belong together, since the whole point of
        // running fixed-depth and timed is to compare them for the same agent.
        config.output_dir = config.heuristic == HeuristicChoice::n1
            ? std::filesystem::path("experiments/results/phase3-learning")
            : std::filesystem::path("experiments/results/phase1-heuristics");
    }

    return config;
}

}  // namespace adversarial_2048
