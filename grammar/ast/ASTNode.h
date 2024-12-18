//
// Created by lasse on 12/18/24.
//

#pragma once

#include <vector>

#include "Range.h"

namespace fsharpgrammar
{
    class IASTNode
    {
    public:
        friend class ASTBuilder;
        virtual ~IASTNode() = default;

        [[nodiscard]] virtual Range get_range() const = 0;
    };

    class AnonModule : public IASTNode
    {
    public:
        AnonModule() : range(Range::create(0, 0)) {}
        ~AnonModule() override = default;

        [[nodiscard]] Range get_range() const override
        {
            return range;
        }
        friend std::string to_string(const AnonModule& moduleOrNamespace)
        {
            return "AnonModule";
        }
    private:
        Range range;
    };

    class Main : public IASTNode
    {
    public:
        Main() : range(Range::create(0, 0)) {}
        ~Main() override = default;

        [[nodiscard]] Range get_range() const override
        {
            return range;
        }

        friend std::string to_string(const Main& main)
        {
            return "Main";
        }

    private:
        std::vector<AnonModule> anon_modules;
        Range range;
    };
} // fsharpmlir
