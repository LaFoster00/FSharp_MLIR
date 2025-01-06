/* Copyright (c) 2012-2017 The ANTLR Project. All rights reserved.
 * Use of this file is governed by the BSD 3-clause license that
 * can be found in the LICENSE.txt file in the project root.
 */

//
//  main.cpp
//  antlr4-cpp-demo
//
//  Created by Mike Lischke on 13.03.16.
//

#include <iostream>
#include <chrono>

#include "antlr4-runtime.h"
#include "FSharpLexer.h"
#include "FSharpParser.h"
#include "utils/Utils.h"

#include "magic_enum/magic_enum.hpp"

#include "ast/AstBuilder.h"
#include "ast/ASTNode.h"
#include "ast/Range.h"

#include "fmt/format.h"
#include "utils/FunctionTimer.h"

using namespace antlr4;
using namespace fsharpgrammar;


int main(int , const char **) {
  std::ifstream file("Program.fs");
  std::stringstream buffer;
  buffer << file.rdbuf();
  std::string fileContent = buffer.str();

  auto start_lexer = std::chrono::high_resolution_clock::now();
  ANTLRInputStream input(fileContent);
  FSharpLexer lexer(&input);
  CommonTokenStream tokens(&lexer);

  tokens.fill();
  auto stop_lexer = std::chrono::high_resolution_clock::now();
  auto duration_lexer = std::chrono::duration_cast<std::chrono::milliseconds>(stop_lexer - start_lexer);
  size_t lastLine = 0;
  for (auto token : tokens.getTokens()) {
    auto type = static_cast<decltype(FSharpLexer::UNKNOWN_CHAR)>(token->getType());
    if (token->getLine() != lastLine)
      std::cout << std::endl << "Line " << token->getLine() << ": \n";
    std::cout << magic_enum::enum_name(type) << ' ';
    lastLine = token->getLine();
  }

  std::cout << "Finished Lexing." << std::endl;

  auto start_parser = std::chrono::high_resolution_clock::now();
  FSharpParser parser(&tokens);
  tree::ParseTree* tree = parser.main();
  auto stop_parser = std::chrono::high_resolution_clock::now();
  auto duration_parser = std::chrono::duration_cast<std::chrono::milliseconds>(stop_parser - start_parser);

  std::cout << tree->toStringTree(&parser, true) << std::endl << std::endl;
  std::cout << "Simplifying tree" << std::endl;

  AstBuilder builder;
  auto start_ast = std::chrono::high_resolution_clock::now();
  auto ast = std::any_cast<ast_ptr<Main>>(builder.visitMain(dynamic_cast<FSharpParser::MainContext*>(tree)));
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

  return 0;
}
