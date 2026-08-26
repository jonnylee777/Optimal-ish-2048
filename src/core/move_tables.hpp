#pragma once

#include <array>
#include <cstddef>
#include <cstdint>

namespace adversarial_2048 {

struct RowMove {
    std::uint16_t row{};
    std::uint32_t score{};
    bool exponent_overflow{};
};

// XOR delta for a vertical move, in packed-board layout. Applying
// `packed ^= (delta << (4 * column))` rewrites one whole column in place, so
// a vertical move needs only one transpose (to read columns as rows) instead
// of transposing, moving horizontally, and transposing back. Technique from
// ronzil/2048-ai-cpp and nneonneo/2048-ai; measured at 2.75x faster per
// vertical move here (see docs/engine-optimization-notes.md).
struct ColumnMove {
    std::uint64_t delta{};
    std::uint32_t score{};
    bool exponent_overflow{};
};

class MoveTables {
public:
    static constexpr std::size_t kRowCount = 1U << 16U;

    MoveTables();

    [[nodiscard]] const RowMove& left(std::uint16_t row) const noexcept;
    [[nodiscard]] const RowMove& right(std::uint16_t row) const noexcept;
    [[nodiscard]] const ColumnMove& up(std::uint16_t column) const noexcept;
    [[nodiscard]] const ColumnMove& down(std::uint16_t column) const noexcept;

private:
    std::array<RowMove, kRowCount> left_{};
    std::array<RowMove, kRowCount> right_{};
    std::array<ColumnMove, kRowCount> up_{};
    std::array<ColumnMove, kRowCount> down_{};
};

[[nodiscard]] const MoveTables& move_tables();

}  // namespace adversarial_2048
