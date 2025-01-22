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

void RunFullTestSuite(InputType inputType, std::string_view fileName, bool emitExe = false, std::optional<std::string> executableOutputPath = std::nullopt)
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
        Action::DumpMLIRFirstLower,
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

    if (!emitExe)
        return;

    FSharpCompiler::compileProgram(
        inputType,
        fileName,
        Action::EmitExecutable,
        true,
        std::move(executableOutputPath)
    );
}

TEST(HelloWorld, BasicAssertion)
{
    RunFullTestSuite(InputType::FSharp, "TestFiles/HelloWorld.fs", true, "HelloWorld");
}

TEST(HelloWorldVariable, BasicAssertion)
{
    RunFullTestSuite(InputType::FSharp, "TestFiles/HelloWorldVariable.fs", true, "HelloWorldVariable");
}

TEST(HelloWorldBranchConstant, BasicAssertion)
{
    RunFullTestSuite(InputType::FSharp, "TestFiles/HelloWorldBranchConstant.fs", false, "HelloWorldBranchConstant");
}

TEST(HelloWorldBranchNestedConstant, BasicAssertion)
{
    RunFullTestSuite(InputType::FSharp, "TestFiles/HelloWorldBranchNestedConstant.fs", false, "HelloWorldBranchNestedConstant");
}

TEST(HelloWorldBranchRelation, BasicAssertion)
{
    RunFullTestSuite(InputType::FSharp, "TestFiles/HelloWorldBranchRelation.fs", false, "HelloWorldBranchRelation");
}

TEST(FunctionDefinition, BasicAssertion)
{
    RunFullTestSuite(InputType::FSharp, "TestFiles/FunctionDefinition.fs");
}

TEST(SimpleAdd, BasicAssertion)
{
    RunFullTestSuite(InputType::FSharp, "TestFiles/SimpleAdd.fs");
}
