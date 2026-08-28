#include "learning/position_store.hpp"

#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string>

namespace adversarial_2048::learning {
namespace {

constexpr char kMagic[8] = {'A', '2', '0', '4', '8', 'P', 'O', 'S'};

}  // namespace

void save_positions(const std::filesystem::path& path, const std::vector<Board>& positions) {
    auto temporary = path;
    temporary += ".tmp";
    {
        std::ofstream stream(temporary, std::ios::binary | std::ios::trunc);
        if (!stream) {
            throw std::runtime_error("cannot open position store for writing: " +
                                     temporary.string());
        }
        stream.write(kMagic, sizeof(kMagic));
        const auto count = static_cast<std::uint64_t>(positions.size());
        stream.write(reinterpret_cast<const char*>(&count), sizeof(count));
        for (const auto& board : positions) {
            stream.write(reinterpret_cast<const char*>(&board.packed_exponents),
                         sizeof(board.packed_exponents));
            stream.write(reinterpret_cast<const char*>(&board.exponent_high_bits),
                         sizeof(board.exponent_high_bits));
        }
        if (!stream) {
            throw std::runtime_error("failed writing position store: " + temporary.string());
        }
    }
    std::filesystem::rename(temporary, path);
}

std::vector<Board> load_positions(const std::filesystem::path& path) {
    std::ifstream stream(path, std::ios::binary);
    if (!stream) {
        throw std::runtime_error("cannot open position store: " + path.string());
    }
    char magic[sizeof(kMagic)]{};
    stream.read(magic, sizeof(magic));
    if (!stream || std::memcmp(magic, kMagic, sizeof(kMagic)) != 0) {
        throw std::runtime_error("not a position store: " + path.string());
    }
    std::uint64_t count = 0;
    stream.read(reinterpret_cast<char*>(&count), sizeof(count));
    if (!stream) {
        throw std::runtime_error("truncated position store: " + path.string());
    }

    std::vector<Board> positions;
    positions.reserve(count);
    for (std::uint64_t index = 0; index < count; ++index) {
        Board board{};
        stream.read(reinterpret_cast<char*>(&board.packed_exponents),
                    sizeof(board.packed_exponents));
        stream.read(reinterpret_cast<char*>(&board.exponent_high_bits),
                    sizeof(board.exponent_high_bits));
        if (!stream) {
            throw std::runtime_error("truncated position store: " + path.string());
        }
        positions.push_back(board);
    }
    return positions;
}

std::vector<ValuedPosition> load_valued_positions(const std::filesystem::path& path) {
    std::ifstream stream(path, std::ios::binary);
    if (!stream) {
        throw std::runtime_error("cannot open target file: " + path.string());
    }
    constexpr char kTargetMagic[8] = {'A', '2', '0', '4', '8', 'T', 'G', 'T'};
    char magic[sizeof(kTargetMagic)]{};
    stream.read(magic, sizeof(magic));
    if (!stream || std::memcmp(magic, kTargetMagic, sizeof(kTargetMagic)) != 0) {
        throw std::runtime_error("not a search-target file: " + path.string());
    }
    std::uint64_t count = 0;
    stream.read(reinterpret_cast<char*>(&count), sizeof(count));
    if (!stream) {
        throw std::runtime_error("truncated target file: " + path.string());
    }

    std::vector<ValuedPosition> out;
    out.reserve(count);
    for (std::uint64_t index = 0; index < count; ++index) {
        ValuedPosition entry{};
        stream.read(reinterpret_cast<char*>(&entry.board.packed_exponents),
                    sizeof(entry.board.packed_exponents));
        stream.read(reinterpret_cast<char*>(&entry.board.exponent_high_bits),
                    sizeof(entry.board.exponent_high_bits));
        stream.read(reinterpret_cast<char*>(&entry.target), sizeof(entry.target));
        if (!stream) {
            throw std::runtime_error("truncated target file: " + path.string());
        }
        out.push_back(entry);
    }
    return out;
}

}  // namespace adversarial_2048::learning
