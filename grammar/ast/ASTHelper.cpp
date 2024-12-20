//
// Created by lasse on 12/19/24.
//

#include "ASTHelper.h"

#include <ranges>

namespace fsharpgrammar::ast
{
    std::string to_string(FSharpParser::Long_identContext* context)
    {
        std::stringstream ss{""};
        bool first = true;
        for (const auto ident : context->ident())
        {
            if (!first)
                ss << '.';
            ss << to_string(ident);
            first = false;
        }
        return ss.str();
    }

    std::string to_string(FSharpParser::IdentContext* context)
    {
        std::stringstream ss{""};
        ss << context->IDENT()->getText();
        return ss.str();
    }
}
