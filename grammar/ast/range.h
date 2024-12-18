//
// Created by lasse on 12/18/24.
//

#pragma once
#include <algorithm>
#include <cstdint>
#include <memory>
#include <string>

#include "utils/Utils.h"

namespace fsharpgrammar
{
    constexpr int64_t pown64(int32_t n)
    {
        if (n == 0)
            return 0L;
        else
            return pown64(n - 1) | (1L << (n - 1));
    }

    constexpr int64_t mask64(int32_t m, int32_t n)
    {
        return pown64(n) << m;
    }

    static constexpr int32_t ColumnBitCount = 22;
    static constexpr int32_t LineBitCount = 31;
    static constexpr int32_t PosBitCount = ColumnBitCount + LineBitCount;
    static constexpr int64_t PosColumnMask = mask64(0, ColumnBitCount);
    static constexpr int64_t LineColumnMask = mask64(ColumnBitCount, LineBitCount);

    struct position {
        static inline int Encoding_size = PosBitCount;

        static constexpr position create(int64_t code)
        {
            return position(code);
        }

        static constexpr position create(int line, int column)
        {
            line = std::max(0, line);
            column = std::max(0, column);
            int64_t code =
                (static_cast<int64_t>(column) & PosColumnMask)
                | ((static_cast<int64_t>(line) << ColumnBitCount) & LineColumnMask);
            return position(code);
        }

        [[nodiscard]] constexpr int line() const
        {
            return static_cast<int>(Encoding >> ColumnBitCount);
        }

        [[nodiscard]] constexpr int column() const
        {
            return static_cast<int>(Encoding & PosColumnMask);
        }

        constexpr bool operator==(const position& other_position) const
        {
            return Encoding == other_position.Encoding;
        }

        [[nodiscard]] constexpr bool is_adjacent_to(const position& other_position) const
        {
            return line() == other_position.line() && column() + 1 == other_position.column();
        }

        // Uses utils to_string to convert the position to a string
        friend std::string to_string(const position& pos)
        {
            return std::to_string(pos.line()) + ":" + std::to_string(pos.column());
        }

    private:
        constexpr explicit position(int64_t code):
        Encoding(code)
#ifdef DEBUG
        , Line(line()), Column(column())
#endif
        {
        }

    public:
        const int64_t Encoding;
#ifdef DEBUG
        const int32_t Line;
        const int32_t Column;
#endif
    };
}

namespace std
{
    template<>
    struct hash<fsharpgrammar::position>
    {
        size_t operator()(const fsharpgrammar::position& pos) const noexcept
        {
            return hash<int64_t>()(pos.Encoding);
        }
    };
}

namespace fsharpgrammar {
    static constexpr int32_t StartColumnBitCount = ColumnBitCount;
    static constexpr int32_t EndColumnBitCount = ColumnBitCount;
    static constexpr int32_t StartLineBitCount = LineBitCount;
    static constexpr int32_t HeightBitCount = 27;
    static constexpr int32_t StartColumnShift = 20;
    static constexpr int32_t EndColumnShift = 42;
    static constexpr int32_t StartLineShift = 0;
    static constexpr int32_t HeightShift = 43;
    static constexpr int64_t StartLineMask = mask64(StartLineShift, StartLineBitCount);
    static constexpr int64_t StartColumnMask = mask64(StartColumnShift, StartColumnBitCount);
    static constexpr int64_t HeightMask = mask64(HeightShift, HeightBitCount);
    static constexpr int64_t EndColumnMask = mask64(EndColumnShift, EndColumnBitCount);

    struct range {
        static constexpr range create(int start_line, int start_column, int end_line, int end_column)
        {
            int64_t code1 =
                ((static_cast<int64_t>(start_column) << StartColumnShift) & StartColumnMask)
                | ((static_cast<int64_t>(end_column) << EndColumnShift) & EndColumnMask)
            ;

            int64_t code2 =
                ((static_cast<int64_t>(start_line) << StartLineShift) & StartLineMask)
                | ((static_cast<int64_t>(end_line - start_line) << HeightShift) & HeightMask)
            ;

            return range(code1, code2);
        }

        static constexpr range create(const position& start, const position& end)
        {
            return create(start.line(), start.column(), end.line(), end.column());
        }

        static constexpr range create(int64_t code1, int64_t code2)
        {
            return range(code1, code2);
        }

        [[nodiscard]] constexpr int32_t start_line() const
        {
            return static_cast<int32_t>((code2 & StartLineMask) >> StartLineShift);
        }

        [[nodiscard]] constexpr int32_t start_column() const
        {
            return static_cast<int32_t>((code1 & StartColumnMask) >> StartColumnShift);
        }

        [[nodiscard]] constexpr int32_t end_line() const
        {
            return static_cast<int32_t>((code2 & HeightMask) >> HeightShift) + start_line();
        }

        [[nodiscard]] constexpr int32_t end_column() const
        {
            return static_cast<int32_t>((code1 & EndColumnMask) >> EndColumnShift);
        }

        [[nodiscard]] constexpr position start() const
        {
            return position::create(start_line(), start_column());
        }

        [[nodiscard]] constexpr position end() const
        {
            return position::create(end_line(), end_column());
        }

        [[nodiscard]] constexpr range start_range() const
        {
            return range::create(start(), start());
        }

        [[nodiscard]] constexpr range end_range() const
        {
            return range::create(end(), end());
        }

        [[nodiscard]] constexpr bool is_adjacent_to(const range& other_range) const
        {
            return this->end() == other_range.start();
        }

        constexpr bool operator==(const range& other) const
        {
            return code1 == other.code1 & code2 == other.code2;
        }

        friend std::string to_string(const range& range)
        {
            return "(" + utils::to_string(range.start()) + '-' + utils::to_string(range.end()) + ')';
        }

    private:
        constexpr range(int64_t code1, int64_t code2)
            :
            code1(code1),
            code2(code2)
        {
        }

    public:
        const int64_t code1;
        const int64_t code2;
    };

} // fsharpgrammar
