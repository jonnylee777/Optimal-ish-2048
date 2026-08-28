#include "experiments/result_writer.hpp"

#include <chrono>
#include <cctype>
#include <ctime>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <string_view>

namespace adversarial_2048 {
namespace {

struct Timestamp {
    std::string iso_utc;
    std::string filename_utc;
};

[[nodiscard]] Timestamp current_timestamp() {
    const auto now = std::chrono::system_clock::now();
    const auto milliseconds = std::chrono::duration_cast<std::chrono::milliseconds>(
        now.time_since_epoch()) % 1'000;
    const auto time = std::chrono::system_clock::to_time_t(now);
    std::tm utc{};
#if defined(_WIN32)
    gmtime_s(&utc, &time);
#else
    gmtime_r(&time, &utc);
#endif

    std::ostringstream iso;
    iso << std::put_time(&utc, "%Y-%m-%dT%H:%M:%S") << '.'
        << std::setw(3) << std::setfill('0') << milliseconds.count() << 'Z';
    std::ostringstream filename;
    filename << std::put_time(&utc, "%Y%m%dT%H%M%S")
             << std::setw(3) << std::setfill('0') << milliseconds.count() << 'Z';
    return {iso.str(), filename.str()};
}

[[nodiscard]] std::string json_escape(std::string_view text) {
    std::ostringstream escaped;
    for (const auto character : text) {
        switch (character) {
            case '"': escaped << "\\\""; break;
            case '\\': escaped << "\\\\"; break;
            case '\b': escaped << "\\b"; break;
            case '\f': escaped << "\\f"; break;
            case '\n': escaped << "\\n"; break;
            case '\r': escaped << "\\r"; break;
            case '\t': escaped << "\\t"; break;
            default:
                if (static_cast<unsigned char>(character) < 0x20U) {
                    escaped << "\\u" << std::hex << std::setw(4) << std::setfill('0')
                            << static_cast<unsigned int>(
                                   static_cast<unsigned char>(character))
                            << std::dec;
                } else {
                    escaped << character;
                }
        }
    }
    return escaped.str();
}

[[nodiscard]] std::string csv_escape(std::string_view text) {
    if (text.find_first_of(",\"\n\r") == std::string_view::npos) {
        return std::string(text);
    }
    std::string escaped{"\""};
    for (const auto character : text) {
        if (character == '"') {
            escaped += "\"\"";
        } else {
            escaped += character;
        }
    }
    escaped += '"';
    return escaped;
}

[[nodiscard]] std::string safe_filename_component(std::string_view text) {
    std::string safe;
    for (const auto character : text) {
        const auto byte = static_cast<unsigned char>(character);
        safe += std::isalnum(byte) != 0 ? character : '_';
    }
    return safe.empty() ? "experiment" : safe;
}

[[nodiscard]] std::string parameter_summary(const ResultMetadata& metadata) {
    std::ostringstream summary;
    summary << std::setprecision(17);
    for (std::size_t index = 0; index < metadata.evaluator_parameters.size(); ++index) {
        if (index != 0) {
            summary << ';';
        }
        summary << metadata.evaluator_parameters[index].name << '='
                << metadata.evaluator_parameters[index].value;
    }
    return summary.str();
}

void finish_file(std::ofstream& stream, const std::filesystem::path& temporary,
                 const std::filesystem::path& destination) {
    stream.close();
    if (!stream) {
        throw std::runtime_error("failed to write experiment result: " +
                                 destination.string());
    }
    std::filesystem::rename(temporary, destination);
}

}  // namespace

ResultFiles write_experiment_results(
    const ExperimentResult& experiment,
    const ResultMetadata& metadata,
    const std::filesystem::path& output_directory) {
    if (experiment.games.empty()) {
        throw std::invalid_argument("cannot write an experiment without games");
    }

    std::filesystem::create_directories(output_directory);
    const auto timestamp = current_timestamp();
    const auto last_seed = experiment.games.back().seed;
    const auto stem = safe_filename_component(experiment.agent_name) + "_depth" +
                      std::to_string(metadata.search_depth) + "_seeds" +
                      std::to_string(experiment.config.first_seed) + '-' +
                      std::to_string(last_seed) + '_' + timestamp.filename_utc;
    const ResultFiles files{
        output_directory / (stem + ".csv"),
        output_directory / (stem + ".json"),
    };
    auto csv_temporary = files.csv;
    auto json_temporary = files.json;
    csv_temporary += ".tmp";
    json_temporary += ".tmp";

    const auto metrics = experiment.metrics();
    const auto parameters = parameter_summary(metadata);
    const auto search = metadata.search_statistics.value_or(SearchStatistics{});

    std::ofstream csv(csv_temporary);
    if (!csv) {
        throw std::runtime_error("failed to open result file: " + csv_temporary.string());
    }
    csv << "timestamp_utc,git_commit,build_mode,agent,evaluator,evaluator_parameters,"
           "feature_configuration,search_depth,depth_definition,seed_partition,"
           "minimum_path_probability,time_limit_seconds,optimization_configuration,seed,score,moves,"
           "max_tile_exponent,max_tile,"
           "runtime_seconds,aggregate_search_total_nodes,aggregate_search_cache_hits\n";
    csv << std::setprecision(17);
    for (const auto& game : experiment.games) {
        const auto max_tile = game.max_tile_exponent == 0
            ? std::uint64_t{0}
            : std::uint64_t{1} << game.max_tile_exponent;
        csv << csv_escape(timestamp.iso_utc) << ','
            << csv_escape(metadata.git_commit) << ','
            << csv_escape(metadata.build_mode) << ','
            << csv_escape(experiment.agent_name) << ','
            << csv_escape(metadata.evaluator_type) << ','
            << csv_escape(parameters) << ','
            << csv_escape(metadata.feature_configuration) << ','
            << metadata.search_depth << ','
            << csv_escape(metadata.depth_definition) << ','
            << csv_escape(metadata.seed_partition) << ','
            << metadata.minimum_path_probability << ','
            << metadata.time_limit_seconds << ','
            << csv_escape(metadata.optimization_configuration) << ','
            << game.seed << ',' << game.score << ',' << game.moves << ','
            << static_cast<unsigned int>(game.max_tile_exponent) << ',' << max_tile << ','
            << game.runtime_seconds << ',' << search.total_nodes() << ','
            << search.cache_hits << '\n';
    }
    finish_file(csv, csv_temporary, files.csv);

    std::ofstream json(json_temporary);
    if (!json) {
        throw std::runtime_error("failed to open result file: " + json_temporary.string());
    }
    json << std::setprecision(17)
         << "{\n"
         << "  \"schema_version\": 1,\n"
         << "  \"timestamp_utc\": \"" << json_escape(timestamp.iso_utc) << "\",\n"
         << "  \"git_commit\": \"" << json_escape(metadata.git_commit) << "\",\n"
         << "  \"build_mode\": \"" << json_escape(metadata.build_mode) << "\",\n"
         << "  \"agent\": \"" << json_escape(experiment.agent_name) << "\",\n"
         << "  \"evaluator\": {\n"
         << "    \"type\": \"" << json_escape(metadata.evaluator_type) << "\",\n"
         << "    \"feature_configuration\": \""
         << json_escape(metadata.feature_configuration) << "\",\n"
         << "    \"parameters\": {";
    for (std::size_t index = 0; index < metadata.evaluator_parameters.size(); ++index) {
        if (index != 0) {
            json << ',';
        }
        json << "\n      \"" << json_escape(metadata.evaluator_parameters[index].name)
             << "\": " << metadata.evaluator_parameters[index].value;
    }
    if (!metadata.evaluator_parameters.empty()) {
        json << '\n' << "    ";
    }
    json << "}\n  },\n"
         << "  \"search\": {\n"
         << "    \"depth\": " << metadata.search_depth << ",\n"
         << "    \"depth_definition\": \"" << json_escape(metadata.depth_definition)
         << "\",\n"
         << "    \"minimum_path_probability\": "
         << metadata.minimum_path_probability << ",\n"
         << "    \"time_limit_seconds\": " << metadata.time_limit_seconds << ",\n"
         << "    \"adaptive_depth\": "
         << (metadata.adaptive_depth_usage.has_value() ? "true" : "false") << ",\n";
    if (metadata.adaptive_depth_usage.has_value()) {
        const auto& usage = *metadata.adaptive_depth_usage;
        json << "    \"adaptive_depth_usage\": {\n"
             << "      \"depth_4_moves\": " << usage[0] << ",\n"
             << "      \"depth_6_moves\": " << usage[1] << ",\n"
             << "      \"depth_8_moves\": " << usage[2] << "\n"
             << "    },\n";
    }
    if (metadata.completed_depth_usage.has_value()) {
        const auto& usage = *metadata.completed_depth_usage;
        json << "    \"completed_depth_usage\": {";
        bool first = true;
        for (std::size_t depth = 0; depth < usage.size(); ++depth) {
            if (usage[depth] == 0) {
                continue;
            }
            json << (first ? "\n" : ",\n")
                 << "      \"depth_" << depth << "_moves\": " << usage[depth];
            first = false;
        }
        if (!first) {
            json << '\n' << "    ";
        }
        json << "},\n";
    }
    if (metadata.deadline_hit_rate.has_value()) {
        json << "    \"deadline_hit_rate\": " << *metadata.deadline_hit_rate << ",\n";
    }
    json
         << "    \"player_nodes\": " << search.player_nodes << ",\n"
         << "    \"chance_nodes\": " << search.chance_nodes << ",\n"
         << "    \"leaf_evaluations\": " << search.leaf_evaluations << ",\n"
         << "    \"spawn_outcomes\": " << search.spawn_outcomes << ",\n"
         << "    \"cache_lookups\": " << search.cache_lookups << ",\n"
         << "    \"cache_hits\": " << search.cache_hits << ",\n"
         << "    \"cache_hit_rate\": " << search.cache_hit_rate() << ",\n"
         << "    \"elapsed_seconds\": " << search.elapsed_seconds << ",\n"
         << "    \"nodes_per_second\": " << search.nodes_per_second() << "\n"
         << "  },\n"
         << "  \"experiment\": {\n"
         << "    \"seed_partition\": \"" << json_escape(metadata.seed_partition) << "\",\n"
         << "    \"first_seed\": " << experiment.config.first_seed << ",\n"
         << "    \"last_seed\": " << last_seed << ",\n"
         << "    \"game_count\": " << experiment.games.size() << ",\n"
         << "    \"worker_threads\": " << experiment.worker_count << ",\n"
         << "    \"timing_valid\": " << (experiment.worker_count == 1 ? "true" : "false") << ",\n"
         << "    \"optimization_configuration\": \""
         << json_escape(metadata.optimization_configuration) << "\"\n"
         << "  },\n"
         << "  \"metrics\": {\n"
         << "    \"mean_score\": " << metrics.mean_score << ",\n"
         << "    \"median_score\": " << metrics.median_score << ",\n"
         << "    \"score_standard_deviation\": "
         << metrics.score_standard_deviation << ",\n"
         << "    \"score_confidence_95_low\": "
         << metrics.score_confidence_95_low << ",\n"
         << "    \"score_confidence_95_high\": "
         << metrics.score_confidence_95_high << ",\n"
         << "    \"best_score\": " << metrics.best_score << ",\n"
         << "    \"worst_score\": " << metrics.worst_score << ",\n"
         << "    \"mean_max_tile\": " << metrics.mean_max_tile << ",\n"
         << "    \"highest_tile\": " << metrics.highest_tile << ",\n"
         << "    \"mode_max_tile\": " << metrics.mode_max_tile << ",\n"
         << "    \"achievement_rate_1024\": "
         << metrics.achievement_rates.tile_1024 << ",\n"
         << "    \"achievement_rate_2048\": "
         << metrics.achievement_rates.tile_2048 << ",\n"
         << "    \"achievement_rate_4096\": "
         << metrics.achievement_rates.tile_4096 << ",\n"
         << "    \"achievement_rate_8192\": "
         << metrics.achievement_rates.tile_8192 << ",\n"
         << "    \"achievement_rate_16384\": "
         << metrics.achievement_rates.tile_16384 << ",\n"
         << "    \"achievement_rate_32768\": "
         << metrics.achievement_rates.tile_32768 << ",\n"
         << "    \"achievement_rate_65536\": "
         << metrics.achievement_rates.tile_65536 << ",\n"
         << "    \"mean_moves\": " << metrics.mean_moves << ",\n"
         << "    \"median_moves\": " << metrics.median_moves << ",\n"
         << "    \"mean_runtime_seconds\": " << metrics.mean_runtime_seconds << ",\n"
         << "    \"mean_milliseconds_per_move\": "
         << metrics.mean_milliseconds_per_move << ",\n"
         << "    \"max_tile_distribution\": {";
    {
        bool first_bucket = true;
        for (std::size_t exponent = 0; exponent < metrics.max_tile_distribution.size(); ++exponent) {
            const auto count = metrics.max_tile_distribution[exponent];
            if (count == 0) {
                continue;
            }
            json << (first_bucket ? "\n" : ",\n")
                 << "      \"" << exponent << "\": " << count;
            first_bucket = false;
        }
        if (!first_bucket) {
            json << '\n' << "    ";
        }
    }
    json << "}\n"
         << "  },\n"
         << "  \"games\": [\n";
    for (std::size_t index = 0; index < experiment.games.size(); ++index) {
        const auto& game = experiment.games[index];
        const auto max_tile = game.max_tile_exponent == 0
            ? std::uint64_t{0}
            : std::uint64_t{1} << game.max_tile_exponent;
        json << "    {\"seed\": " << game.seed
             << ", \"score\": " << game.score
             << ", \"moves\": " << game.moves
             << ", \"max_tile_exponent\": "
             << static_cast<unsigned int>(game.max_tile_exponent)
             << ", \"max_tile\": " << max_tile
             << ", \"runtime_seconds\": " << game.runtime_seconds << '}';
        json << (index + 1U == experiment.games.size() ? "\n" : ",\n");
    }
    json << "  ]\n}\n";
    finish_file(json, json_temporary, files.json);
    return files;
}

}  // namespace adversarial_2048
