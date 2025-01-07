//
// Created by lasse on 06/01/2025.
//

#pragma once

#include <string>

#include "ast/ASTNode.h"

namespace fsharpgrammar
{
    class Grammar
    {
    public:
        static std::unique_ptr<ast::Main> parse(std::string_view source,
                                             bool print_lexer_output = true,
                                             bool print_parser_output = true,
                                             bool print_ast_output = true);
    };
} // fsharpgrammar
