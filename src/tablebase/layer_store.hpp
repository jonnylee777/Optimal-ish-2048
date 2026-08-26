#pragma once

#include "tablebase/solver.hpp"

#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

namespace adversarial_2048::tablebase {

// Disk-backed layer storage, so a table never needs to be resident all at once.
//
// Two file kinds per layer:
//   <prefix>_<layer>.boards  — sorted uint64 boards, 8 bytes each (generation)
//   <prefix>_<layer>.solved  — sorted {uint64 board, double probability}
//                              records, 16 bytes each (backward solve)
//
// The reference project quantizes probabilities to uint32 scaled by 4e9 to save
// a third of the space (`SuccessEntry<uint32_t>`). We keep full doubles so the
// disk path reproduces the in-memory solver bit-for-bit, which is what lets one
// validate the other. Switch to quantized storage if disk ever becomes the
// binding constraint.
class LayerStore {
public:
    LayerStore(std::filesystem::path directory, std::string prefix);

    [[nodiscard]] std::filesystem::path boards_path(std::size_t layer) const;
    [[nodiscard]] std::filesystem::path solved_path(std::size_t layer) const;

    void write_boards(std::size_t layer, const std::vector<std::uint64_t>& sorted_boards) const;
    [[nodiscard]] std::vector<std::uint64_t> read_boards(std::size_t layer) const;

    void write_solved(std::size_t layer, const SolvedLayer& solved) const;
    [[nodiscard]] SolvedLayer read_solved(std::size_t layer) const;

    [[nodiscard]] bool has_solved(std::size_t layer) const;
    void remove_boards(std::size_t layer) const;

    [[nodiscard]] std::uintmax_t total_bytes() const;
    void remove_all() const;

private:
    std::filesystem::path directory_;
    std::string prefix_;
};

}  // namespace adversarial_2048::tablebase
