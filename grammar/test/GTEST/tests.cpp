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

#define AST_GENERATION_TEST(Name, Type, SourceCode) \
TEST(Name, Type) \
{ \
    antlr4::ANTLRInputStream input(SourceCode); \
    fsharpgrammar::FSharpLexer lexer(&input);\
    antlr4::CommonTokenStream tokens(&lexer);\
    \
    tokens.fill();\
    size_t lastLine = 0;\
    for (auto token : tokens.getTokens())\
    {\
        auto type = static_cast<decltype(fsharpgrammar::FSharpLexer::UNKNOWN_CHAR)>(token->getType());\
        if (token->getLine() != lastLine)\
            std::cout << std::endl << "Line " << token->getLine() << ": \n";\
        std::cout << magic_enum::enum_name(type) << ' ';\
        lastLine = token->getLine();\
    }\
    \
    std::cout << "Finished Lexing." << std::endl;\
    \
    fsharpgrammar::FSharpParser parser(&tokens);\
    antlr4::tree::ParseTree* tree = parser.main();\
    \
    /* std::cout << tree->toStringTree(&parser, true) << std::endl << std::endl; */ \
    std::cout << "Simplifying tree" << std::endl;\
    \
    fsharpgrammar::AstBuilder builder;\
    auto ast = std::any_cast<fsharpgrammar::ast_ptr<fsharpgrammar::Main>>(builder.visitMain(dynamic_cast<fsharpgrammar::FSharpParser::MainContext*>(tree)));\
    std::string ast_string = utils::to_string(*ast);\
    \
    fmt::print("AST Generation Result = \n{}\n", *ast);\
}

namespace basic_statements
{
    constexpr std::string_view basic_statements = R"(
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
    AST_GENERATION_TEST(SimpleExpressions, BasicAssertions, basic_statements)
}

namespace ast_printer_source
{
    constexpr std::string_view ast_printer_src = R"###(
open System.IO
open FSharp.Compiler.CodeAnalysis
open FSharp.Compiler.Text
open FSharp.Compiler.Syntax
open FSharp.Compiler.Tokenization

a | b & 10 = d != e < f <= g > h >= i + j - k * l / m % n
a.[10]

let a: int = 10 * 10
let b =
    let c =
        10 + 10
    c + 10
let c: float = float(10 + 20 * 30 - 50)

let d = if 10 > 20 then 30 else 40
let e =
    if 10 > 20 then
        30
    else
        40

let f x y z = x + y + z
let g = f 10 e 30


let getSourceTokenizer (file, input) =
    let sourceTok = FSharpSourceTokenizer([], file, None, Some true)
    let tokenizer = sourceTok.CreateLineTokenizer(input)
    tokenizer

 /// Tokenize a single line of F# code
let rec tokenizeLine (tokenizer:FSharpLineTokenizer) state =
    match tokenizer.ScanToken(state) with
    | Some tok, state ->
        // Print token name
        printf "%s " tok.TokenName
        // Tokenize the rest, in the new state
        tokenizeLine tokenizer state
    | None, state -> state

/// Print token names for multiple lines of code
let rec tokenizeLines (sourceTok: FSharpSourceTokenizer) state count lines  =
    match lines with
    | line::lines ->
        // Create tokenizer & tokenize single line
        printfn "\nLine %d" count
        let tokenizer = sourceTok.CreateLineTokenizer(line)
        let state = tokenizeLine tokenizer state
        // Tokenize the rest using new state
        tokenizeLines sourceTok state (count+1) lines
    | [] -> ()

/// Get untyped tree for a specified input
let getUntypedTree (file, input) =
    let checker = FSharpChecker.Create()
    let inputSource = SourceText.ofString input
    // Get compiler options for the 'project' implied by a single script file
    let projOptions, diagnostics =
        checker.GetProjectOptionsFromScript(file, inputSource, assumeDotNetFramework=false)
        |> Async.RunSynchronously

    let parsingOptions, _errors = checker.GetParsingOptionsFromProjectOptions(projOptions)

    // Run the first phase (untyped parsing) of the compiler
    let parseFileResults =
        checker.ParseFile(file, inputSource, parsingOptions)
        |> Async.RunSynchronously

    parseFileResults.ParseTree


let printAst (ast: ParsedInput) =
    match ast with
    | ParsedInput.ImplFile parsedImplFileInput ->
        printfn "Implementation File: %A" parsedImplFileInput
    | ParsedInput.SigFile parsedSigFileInput ->
        printfn "Signature File: %A" parsedSigFileInput


let readFileAsString (filePath: string) : string =
    File.ReadAllText(filePath)

// Example usage
let filePath = "Program.fs"
let fileContents = readFileAsString filePath

let lines = fileContents.Split('\r', '\n')

let sourceTok = FSharpSourceTokenizer([], Some "Test.fs", None, Some true)

lines
|> List.ofSeq
|> tokenizeLines sourceTok FSharpTokenizerLexState.Initial 1

// Inspect the syntax tree
getUntypedTree (filePath, fileContents)
|> printAst

//Pattern matching
let tuple_pat = (1, "hello", true)
match tuple_pat with
| (1, "hello", true) -> printfn "Matched tuple (1, \"hello\", true)"
| _ -> printfn "No match"

let and_pat = (1, "hello", 3)
match and_pat with
| (1, "hello", 3) & (1, _, _) -> printfn "Matched first part and second part"
| _ -> printfn "No match"

let or_pat = "apple"
match or_pat with
| "apple" | "banana" -> printfn "Matched apple or banana"
| _ -> printfn "No match"

let as_pat = Some(10)
match as_pat with
| Some x as value -> printfn "Matched Some with value: %d, full match: %A" x value
| None -> printfn "Matched None"

let cons_pat = [1; 2; 3]
match cons_pat with
| 1 :: tail -> printfn "Matched 1 as head, and tail is: %A" tail
| _ -> printfn "No match"

if 10 < 20 then printfn "Hello World"

)###";
    AST_GENERATION_TEST(AST_PRINTER, BasicAssertions, ast_printer_src)
}