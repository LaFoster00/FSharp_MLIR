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

#include "antlr4-runtime.h"
#include "FSharpLexer.h"
#include "FSharpParser.h"
#include "utils/Utils.h"

#include "magic_enum/magic_enum.hpp"

#include "ast/AstBuilder.h"
#include "ast/ASTNode.h"
#include "ast/Range.h"

#include "fmt/format.h"

using namespace antlr4;
using namespace fsharpgrammar;


int main(int , const char **) {
  std::ifstream file("Program.fs");
  std::stringstream buffer;
  buffer << file.rdbuf();
  std::string fileContent = buffer.str();

  ANTLRInputStream input(fileContent);
  FSharpLexer lexer(&input);
  CommonTokenStream tokens(&lexer);

  tokens.fill();
  size_t lastLine = 0;
  for (auto token : tokens.getTokens()) {
    auto type = static_cast<decltype(FSharpLexer::UNKNOWN_CHAR)>(token->getType());
    if (token->getLine() != lastLine)
      std::cout << std::endl << "Line " << token->getLine() << ": \n";
    std::cout << magic_enum::enum_name(type) << ' ';
    lastLine = token->getLine();
  }

  std::cout << "Finished Lexing." << std::endl;

  FSharpParser parser(&tokens);
  tree::ParseTree* tree = parser.main();

  std::cout << tree->toStringTree(&parser, true) << std::endl << std::endl;

  std::cout << "Simplifying tree" << std::endl;

  AstBuilder builder;
  std::any simplified_tree = builder.visitMain(dynamic_cast<FSharpParser::MainContext*>(tree));
  std::cout << tree->toStringTree(&parser, true) << std::endl;

  ModuleOrNamespace bla(ModuleOrNamespace::Type::NamedModule, "Hello World", std::vector<ast_ptr<ModuleDeclaration>>{}, Range::create(30, 10, 2, 3));

  std::cout << utils::to_string(bla) << std::endl;
  fmt::print("Test Node {}", bla);

  return 0;
}
