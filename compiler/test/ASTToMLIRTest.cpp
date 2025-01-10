//
// Created by lasse on 1/7/25.
//

#include "compiler/ASTToMLIR.h"

#include <chrono>
#include <iostream>
#include <thread>

#include <gtest/gtest.h>

#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/Target/LLVMIR/ModuleImport.h"

using namespace fsharpgrammar::compiler;
using namespace std::chrono_literals;

mlir::OwningOpRef<mlir::ModuleOp> generate_mlir(std::string_view source,
                                                mlir::MLIRContext& context)
{
    return MLIRGen::mlirGen(
        context,
        source
        );
}

#define GENERATE_AND_DUMP_MLIR(source) \
    mlir::MLIRContext context; \
    auto result = generate_mlir(source, context); \
    std::cout << std::flush; \
    std::this_thread::sleep_for(100ms); \
    result->dump()

TEST(HelloWorld, BasicAssertion)
{
    GENERATE_AND_DUMP_MLIR(
        R"(
printfn 1
)"
    );
}

TEST(SimpleNamedModule, BasicAssertion)
{
    GENERATE_AND_DUMP_MLIR(
        R"(
module outer

printfn 1
)"
    );
}

TEST(SimpleNestedModule, BasicAssertion)
{
    GENERATE_AND_DUMP_MLIR(
        R"(
module outer
printfn 1
module nested =
    printfn 2
)"
    );
}

TEST(MultiNestedModule, BasicAssertion)
{
    GENERATE_AND_DUMP_MLIR(
        R"(
module outer
printfn 1
module nested =
    printfn 2
    module nested_nested =
        printfn 3
module nested2 =
    printfn 4
)"
    );
}
