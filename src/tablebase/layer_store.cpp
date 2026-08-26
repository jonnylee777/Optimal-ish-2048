#include "tablebase/layer_store.hpp"

#include <cstdio>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <utility>

namespace adversarial_2048::tablebase {
namespace {

constexpr std::size_t kBoardRecordBytes = sizeof(std::uint64_t);            // 8
constexpr std::size_t kSolvedRecordBytes = sizeof(std::uint64_t) + sizeof(double);  // 16

void write_all(const std::filesystem::path& path, const char* data, std::size_t bytes) {
    // Write to a temporary then rename, so an interrupted build never leaves a
    // half-written layer that a later run would happily read as complete.
    auto temporary = path;
    temporary += ".tmp";
    {
        std::ofstream stream(temporary, std::ios::binary | std::ios::trunc);
        if (!stream) {
            throw std::runtime_error("cannot open layer file: " + temporary.string());
        }
        if (bytes > 0) {
            stream.write(data, static_cast<std::streamsize>(bytes));
        }
        stream.flush();
        if (!stream) {
            throw std::runtime_error("failed writing layer file: " + temporary.string());
        }
    }
    std::filesystem::rename(temporary, path);
}

[[nodiscard]] std::vector<char> read_all(const std::filesystem::path& path) {
    std::ifstream stream(path, std::ios::binary | std::ios::ate);
    if (!stream) {
        throw std::runtime_error("cannot open layer file: " + path.string());
    }
    const auto size = static_cast<std::streamoff>(stream.tellg());
    stream.seekg(0);
    std::vector<char> buffer(static_cast<std::size_t>(size));
    if (size > 0) {
        stream.read(buffer.data(), size);
        if (!stream) {
            throw std::runtime_error("failed reading layer file: " + path.string());
        }
    }
    return buffer;
}

}  // namespace

LayerStore::LayerStore(std::filesystem::path directory, std::string prefix)
    : directory_(std::move(directory)), prefix_(std::move(prefix)) {
    std::filesystem::create_directories(directory_);
}

std::filesystem::path LayerStore::boards_path(std::size_t layer) const {
    return directory_ / (prefix_ + "_" + std::to_string(layer) + ".boards");
}

std::filesystem::path LayerStore::solved_path(std::size_t layer) const {
    return directory_ / (prefix_ + "_" + std::to_string(layer) + ".solved");
}

void LayerStore::write_boards(
    std::size_t layer, const std::vector<std::uint64_t>& sorted_boards) const {
    write_all(boards_path(layer), reinterpret_cast<const char*>(sorted_boards.data()),
              sorted_boards.size() * kBoardRecordBytes);
}

std::vector<std::uint64_t> LayerStore::read_boards(std::size_t layer) const {
    const auto path = boards_path(layer);
    if (!std::filesystem::exists(path)) {
        return {};
    }
    const auto bytes = read_all(path);
    if (bytes.size() % kBoardRecordBytes != 0) {
        throw std::runtime_error("corrupt boards layer: " + path.string());
    }
    std::vector<std::uint64_t> boards(bytes.size() / kBoardRecordBytes);
    if (!boards.empty()) {
        std::memcpy(boards.data(), bytes.data(), bytes.size());
    }
    return boards;
}

void LayerStore::write_solved(std::size_t layer, const SolvedLayer& solved) const {
    if (solved.boards.size() != solved.probabilities.size()) {
        throw std::runtime_error("solved layer has mismatched board/probability counts");
    }
    std::vector<char> buffer(solved.boards.size() * kSolvedRecordBytes);
    for (std::size_t index = 0; index < solved.boards.size(); ++index) {
        auto* record = buffer.data() + index * kSolvedRecordBytes;
        std::memcpy(record, &solved.boards[index], sizeof(std::uint64_t));
        std::memcpy(record + sizeof(std::uint64_t), &solved.probabilities[index], sizeof(double));
    }
    write_all(solved_path(layer), buffer.data(), buffer.size());
}

SolvedLayer LayerStore::read_solved(std::size_t layer) const {
    SolvedLayer solved;
    const auto path = solved_path(layer);
    if (!std::filesystem::exists(path)) {
        return solved;
    }
    const auto bytes = read_all(path);
    if (bytes.size() % kSolvedRecordBytes != 0) {
        throw std::runtime_error("corrupt solved layer: " + path.string());
    }
    const auto count = bytes.size() / kSolvedRecordBytes;
    solved.boards.resize(count);
    solved.probabilities.resize(count);
    for (std::size_t index = 0; index < count; ++index) {
        const auto* record = bytes.data() + index * kSolvedRecordBytes;
        std::memcpy(&solved.boards[index], record, sizeof(std::uint64_t));
        std::memcpy(&solved.probabilities[index], record + sizeof(std::uint64_t), sizeof(double));
    }
    return solved;
}

bool LayerStore::has_solved(std::size_t layer) const {
    return std::filesystem::exists(solved_path(layer));
}

void LayerStore::remove_boards(std::size_t layer) const {
    std::error_code error;
    std::filesystem::remove(boards_path(layer), error);
}

std::uintmax_t LayerStore::total_bytes() const {
    std::uintmax_t total = 0;
    std::error_code error;
    for (const auto& entry : std::filesystem::directory_iterator(directory_, error)) {
        if (entry.is_regular_file(error)) {
            total += entry.file_size(error);
        }
    }
    return total;
}

void LayerStore::remove_all() const {
    std::error_code error;
    std::filesystem::remove_all(directory_, error);
}

}  // namespace adversarial_2048::tablebase
