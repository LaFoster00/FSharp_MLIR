//
// Created by lasse on 1/3/25.
//

#include "antlr4-runtime.h"
#include "FSharpLexer.h"
#include "FSharpParser.h"

#include <fstream>
#include <gtest/gtest.h>

#include "utils/Utils.h"
#include "utils/FunctionTimer.h"
#include "magic_enum/magic_enum.hpp"

#include "ast/AstBuilder.h"
#include "ast/ASTNode.h"

// Demonstrate some basic assertions.
TEST(HelloTest, BasicAssertions)
{
    constexpr std::string_view input_string =R"(
let i =
    let mutable z = 5 * (a 10 b)
    z <- z * (c 20); z * 2

let a = 10
let c = 2 * -10
let d = 3 * 10 * a
let e = 4 * (a 10)
let f = 5 * (a 10 b c 10)
let g = 6 * ((a 10) + 10)
let h = 5 * (a 10 b) * (c 20)


printf b
printf a a (foo 10) 10 20

module test =
    bla (10, 20, dergrößte)
    bla (10, 20, dergrößte (foo 10))
    //NOTE: If dergrößte is a function, it should be called in parantheses otherwise it will be treated an id to a value
    bla 10 20 dergrößte (foo 10)

    let foo x = x * x
    let foo (x: int) : int = x * x
    let foo x =
        x * x
        x * x

module test2 =
    let foo x = x * x
    let foo (x: int) : int = x * x
    let foo x =
      x * x

//Example Curring:
let multiply x y = x * y
let double x = multiply 2
let result = double 5 // result is 10

//Example Typing:
//Basic Types
let intValue: int = 42
let floatValue: float = 3.14
let boolValue: bool = true
let charValue: char = 'A'
let stringValue: string = "Hello, F#"

//Type Inference
let inferredInt = 42  // inferred as int
let inferredString = "Hello"  // inferred as string

//Function Types
let square (x: int) : int = x * x // int -> int

//Constant matching:
let describeNumber x =
    match x with
    | 0 -> "Zero"
    | 1 -> "One"
    | 2 -> "Two"
    | _ -> "Other"

let result1 = describeNumber 0  // result1 is "Zero"
let result2 = describeNumber 1  // result2 is "One"
let result3 = describeNumber 3  // result3 is "Other"
)";

    auto start_lexer = std::chrono::high_resolution_clock::now();
    antlr4::ANTLRInputStream input(input_string);
    fsharpgrammar::FSharpLexer lexer(&input);
    antlr4::CommonTokenStream tokens(&lexer);

    tokens.fill();
    auto stop_lexer = std::chrono::high_resolution_clock::now();
    auto duration_lexer = std::chrono::duration_cast<std::chrono::milliseconds>(stop_lexer - start_lexer);
    size_t lastLine = 0;
    for (auto token : tokens.getTokens())
    {
        auto type = static_cast<decltype(fsharpgrammar::FSharpLexer::UNKNOWN_CHAR)>(token->getType());
        if (token->getLine() != lastLine)
            std::cout << std::endl << "Line " << token->getLine() << ": \n";
        std::cout << magic_enum::enum_name(type) << ' ';
        lastLine = token->getLine();
    }

    std::cout << "Finished Lexing." << std::endl;

    auto start_parser = std::chrono::high_resolution_clock::now();
    fsharpgrammar::FSharpParser parser(&tokens);
    antlr4::tree::ParseTree* tree = parser.main();
    auto stop_parser = std::chrono::high_resolution_clock::now();
    auto duration_parser = std::chrono::duration_cast<std::chrono::milliseconds>(stop_parser - start_parser);

    std::cout << tree->toStringTree(&parser, true) << std::endl << std::endl;
    std::cout << "Simplifying tree" << std::endl;

    fsharpgrammar::AstBuilder builder;
    auto start_ast = std::chrono::high_resolution_clock::now();
    auto ast = std::any_cast<fsharpgrammar::ast_ptr<fsharpgrammar::Main>>(builder.visitMain(dynamic_cast<fsharpgrammar::FSharpParser::MainContext*>(tree)));
    auto stop_ast = std::chrono::high_resolution_clock::now();
    auto duration_ast = std::chrono::duration_cast<std::chrono::milliseconds>(stop_ast - start_ast);

    auto start_ast_print = std::chrono::high_resolution_clock::now();
    std::string ast_string = utils::to_string(*ast);
    auto stop_ast_print = std::chrono::high_resolution_clock::now();
    auto duration_ast_print = std::chrono::duration_cast<std::chrono::milliseconds>(stop_ast_print - start_ast_print);

    fmt::print(
        "AST Generation Result = \n{}\n"
        "\n\tAST Generation Time: {}ms"
        "\n\tAST Printing Time: {}ms"
        "\n\tLexing Time: {}ms"
        "\n\tParser Time: {}ms\n",
        *ast,
        duration_ast.count(),
        duration_ast_print.count(),
        duration_lexer.count(),
        duration_parser.count());

    FunctionTimer::PrintTimings();
}
