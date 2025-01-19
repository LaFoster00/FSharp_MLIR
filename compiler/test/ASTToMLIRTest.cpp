//
// Created by lasse on 1/7/25.
//

#include "compiler/Compiler.h"

#include <chrono>
#include <iostream>
#include <thread>
#include <compiler/ASTToMLIR.h>

#include <gtest/gtest.h>

#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/Target/LLVMIR/ModuleImport.h"

using namespace fsharp::compiler;
using namespace std::chrono_literals;

mlir::OwningOpRef<mlir::ModuleOp> generate_mlir(std::string_view source,
                                                mlir::MLIRContext& context)
{
    return fsharpgrammar::compiler::MLIRGen::mlirGen(
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

void RunFullTestSuite(InputType inputType, std::string_view fileName)
{
    FSharpCompiler::compileProgram(
        inputType,
        fileName,
        Action::DumpAST,
        false
    );

    FSharpCompiler::compileProgram(
        inputType,
        fileName,
        Action::DumpMLIR,
        false
    );

    FSharpCompiler::compileProgram(
        inputType,
        fileName,
        Action::DumpMLIRAffine,
        false
    );

    FSharpCompiler::compileProgram(
        inputType,
        fileName,
        Action::DumpMLIRLLVM,
        true
    );

    FSharpCompiler::compileProgram(
        inputType,
        fileName,
        Action::DumpLLVMIR,
        false
    );

    FSharpCompiler::compileProgram(
        inputType,
        fileName,
        Action::RunJIT,
        true
    );
}

TEST(HelloWorld, BasicAssertion)
{
    RunFullTestSuite(InputType::FSharp, "TestFiles/HelloWorld.fs");
}

TEST(HelloWorldVariable, BasicAssertion)
{
    RunFullTestSuite(InputType::FSharp, "TestFiles/HelloWorldVariable.fs");
}

TEST(AritTest, BasicAssertion)
{
    GENERATE_AND_DUMP_MLIR(
        R"(
let a = 1 + 2 + 3 - 3
print a
)"
    );
}

TEST(SimpleAdd, BasicAssertion)
{
    RunFullTestSuite(InputType::FSharp, "TestFiles/SimpleAdd.fs");
}
