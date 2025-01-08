//
// Created by lasse on 1/7/25.
//

#include "generator/ASTToMLIR.h"

#include <gtest/gtest.h>

#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/Target/LLVMIR/ModuleImport.h"

using namespace fsharpgrammar::compiler;

TEST(HelloWorld, BasicAssertion)
{
    mlir::MLIRContext context;
    context.getOrLoadDialect<mlir::BuiltinDialect>();
    auto result = MLIRGen::mlirGen(
        context,
        R"(
printfn "Hello World!"
)"
    );
    result->dump();
}
