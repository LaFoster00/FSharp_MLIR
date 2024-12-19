//
// Created by lasse on 12/19/24.
//

#include "ASTHelper.h"

namespace fsharpgrammar::ast
{


    std::string to_string(FSharpParser::Long_identContext* context)
    {
        std::stringstream ss{""};
        for (const auto ident : context->ident())
            ss << to_string(ident);
        return ss.str();
    }

    std::string to_string(FSharpParser::IdentContext* context)
    {
        std::stringstream ss{""};
        ss << context->IDENT()->getText();
        return ss.str();
    }
}