//
// Created by lasse on 12/19/24.
//

#pragma once

#include "ASTNode.h"
#include "FSharpParser.h"
#include "Range.h"
#include "fmt/format.h"

#include "utils/Utils.h"


namespace fsharpgrammar::ast
{
    template<typename T>
    ast_ptr<T> any_cast(std::any& obj, antlr4::ParserRuleContext *parserRuleContext)
    {
        try
        {
            return std::any_cast<ast_ptr<T>>(obj);
        }
        catch(std::bad_any_cast&)
        {
            std::string error_message = fmt::format(
                "AST Building exception at \"{}\" {} expected {} but got {} instead",
                parserRuleContext->start->getText(),
                utils::to_string(Range::create(parserRuleContext)),
                utils::type_name<T>(),
                utils::demangle(obj.type().name()));
            throw antlr4::ParseCancellationException(error_message);
        }
    }

    template<typename T>
    ast_ptr<T> any_cast(std::any&& obj, antlr4::ParserRuleContext *parserRuleContext)
    {
        try
        {
            return std::any_cast<ast_ptr<T>>(std::move(obj));
        }
        catch(std::bad_any_cast&)
        {
            std::string error_message = fmt::format(
                "AST Building exception at \"{}\" {} expected {} but got {} instead",
                parserRuleContext->start->getText(),
                utils::to_string(Range::create(parserRuleContext)),
                utils::type_name<T>(),
                utils::demangle(obj.type().name()));
            throw antlr4::ParseCancellationException(error_message);
        }
    }

    std::string to_string(FSharpParser::Long_identContext* context);
    std::string to_string(FSharpParser::IdentContext* context);
}

