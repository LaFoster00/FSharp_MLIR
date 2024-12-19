//
// Created by lasse on 12/19/24.
//

#pragma once

#include <format>

#include "FSharpParser.h"
#include "Range.h"

namespace fsharpgrammar
{
    template<typename T>
    using ast_ptr = std::shared_ptr<T>;

    template <typename T, typename... Args>
    auto make_ast(Args&&... args) -> decltype(std::make_shared<T>(std::forward<Args>(args)...)) {
        return std::make_shared<T>(std::forward<Args>(args)...);
    }
}

namespace fsharpgrammar::ast
{
    template<typename T>
    T any_cast_ast(std::any& obj, antlr4::ParserRuleContext *parserRuleContext)
    {
        try
        {
            return std::any_cast<ast_ptr<T>>(obj);
        }
        catch(std::bad_any_cast&)
        {
            std::string error_message = std::format(
                "AST Building exception at \"{}\" {} expected {} but got {} instead",
                parserRuleContext->getText(),
                Range::create(parserRuleContext),
                typeid(T).name(),
                obj.type().name());
            throw antlr4::ParseCancellationException(error_message);
        }
    }

    std::string to_string(FSharpParser::Long_identContext* context);
    std::string to_string(FSharpParser::IdentContext* context);
}

