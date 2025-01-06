//
// Created by lasse on 12/19/24.
//

#pragma once

#include <any>
#include <memory>

#include <antlr4-runtime.h>
#include <fmt/format.h>
#include <cpptrace/cpptrace.hpp>

#include "utils/Utils.h"
#include "Range.h"

namespace fsharpgrammar::ast
{
    template <typename T>
        using ast_ptr = std::shared_ptr<T>;

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
                utils::type_name<std::shared_ptr<T>>(),
                utils::demangle(obj.type().name()));
            auto stacktrace = cpptrace::generate_trace();
            stacktrace.print();
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
            auto stacktrace = cpptrace::generate_trace();
            stacktrace.print();
            throw antlr4::ParseCancellationException(error_message);
        }
    }
}

