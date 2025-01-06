//
// Created by lasse on 06/01/2025.
//

#include "Grammar.h"

#include <antlr4-runtime.h>
#include <FSharpLexer.h>
#include <FSharpParser.h>
#include <magic_enum/magic_enum.hpp>

#include "ast/AstBuilder.h"
#include "utils/FunctionTimer.h"

namespace fsharpgrammar
{
    void lex_source(const bool print_lexer_output, antlr4::CommonTokenStream& tokens)
    {
        tokens.fill();
        if (print_lexer_output)
        {
            size_t lastLine = 0;
            for (auto token : tokens.getTokens())
            {
                auto type = static_cast<decltype(FSharpLexer::UNKNOWN_CHAR)>(token->getType());
                if (token->getLine() != lastLine)
                    std::cout << std::endl << "Line " << token->getLine() << ": \n";
                std::cout << magic_enum::enum_name(type) << ' ';
                lastLine = token->getLine();
            }
        }
    }

    antlr4::tree::ParseTree* parse_source(const bool print_parser_output, FSharpParser& parser)
    {
        antlr4::tree::ParseTree* tree = parser.main();
        if (print_parser_output)
            std::cout << tree->toStringTree(&parser, true) << std::endl << std::endl;
        return tree;
    }

    std::unique_ptr<ast::Main> Grammar::parse(const std::string_view source,
                                              const bool print_lexer_output,
                                              const bool print_parser_output,
                                              const bool print_ast_output)
    {
        antlr4::ANTLRInputStream input(source);
        FSharpLexer lexer(&input);
        antlr4::CommonTokenStream tokens(&lexer);
        lex_source(print_lexer_output, tokens);
        std::cout << "Finished Lexing." << std::endl;

        FSharpParser parser(&tokens);
        antlr4::tree::ParseTree* const tree = parse_source(print_parser_output, parser);
        std::cout << "Finished Parsing" << std::endl;


        ast::AstBuilder builder;
        auto ast = builder.BuildAst(dynamic_cast<FSharpParser::MainContext*>(tree));

        if (print_ast_output)
        {
            std::string ast_string = utils::to_string(*ast);
            fmt::print("AST Generation Result = \n{}\n", *ast);
        }
        return ast;
    }
} // fsharpgrammar
