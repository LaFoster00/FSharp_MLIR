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
  for (auto token : tokens.getTokens()) {
    std::cout << token->toString() << std::endl;
  }

  FSharpParser parser(&tokens);
  tree::ParseTree* tree = parser.main();

  std::cout << tree->toStringTree(&parser, true) << std::endl << std::endl;

  return 0;
}
