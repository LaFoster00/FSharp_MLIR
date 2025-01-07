//
// Created by lasse on 1/7/25.
//

#include "generator/ASTToMLIR.h"

#include <gtest/gtest.h>

TEST(HelloWorld, BasicAssertion)
{
    mlir::MLIRContext context;
    generateMLIRFromAST(R"(
printfn "Hello World!"
)", context);
}