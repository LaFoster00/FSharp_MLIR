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

mlir::OwningOpRef<mlir::ModuleOp> generate_mlir(std::string_view source,
                                                mlir::MLIRContext& context)
{
    context.getOrLoadDialect<mlir::BuiltinDialect>();

    return MLIRGen::mlirGen(
        context,
        source
    );
}

#define GENERATE_AND_DUMP_MLIR(source) \
    mlir::MLIRContext context; \
    auto result = generate_mlir(source, context); \
    result->dump()

TEST(HelloWorld, BasicAssertion)
{
    GENERATE_AND_DUMP_MLIR(
        R"(
printfn "Hello World!"
)"
    );
}

TEST(SimpleNamedModule, BasicAssertion)
{
    GENERATE_AND_DUMP_MLIR(
        R"(
module outer

printfn "Hello World!"
)"
    );
}

TEST(SimpleNestedModule, BasicAssertion)
{
    GENERATE_AND_DUMP_MLIR(
        R"(
module outer
printfn "Outer"
module nested =
    printfn "Inner"
)"
    );
}
