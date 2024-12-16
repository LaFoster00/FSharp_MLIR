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

#include "magic_enum/magic_enum.hpp"

using namespace antlr4;
using namespace fsharpgrammar;

int main(int , const char **) {
  std::ifstream file("SimpleExpression.fs");
  std::stringstream buffer;
  buffer << file.rdbuf();
  std::string fileContent = buffer.str();

  ANTLRInputStream input(fileContent);
  FSharpLexer lexer(&input);
  CommonTokenStream tokens(&lexer);

  tokens.fill();
  size_t lastLine = 0;
  for (auto token : tokens.getTokens()) {
    auto type = static_cast<decltype(FSharpLexer::NUMBER)>(token->getType());
    if (token->getLine() != lastLine)
      std::cout << std::endl << "Line " << token->getLine() << ": \n";
    std::cout << magic_enum::enum_name(type) << ' ';
    lastLine = token->getLine();
  }

  std::cout << "Finished Parsing." << std::endl;

  FSharpParser parser(&tokens);
  tree::ParseTree* tree = parser.main();

  std::cout << tree->toStringTree(&parser, true) << std::endl << std::endl;

  return 0;
}
