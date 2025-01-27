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

int RunFullTestSuite(InputType inputType, std::string_view fileName, bool emitExe = false,
                     std::optional<std::string> executableOutputPath = std::nullopt)
{
    int result = 0;
    result = FSharpCompiler::compileProgram(
        inputType,
        fileName,
        Action::DumpAST,
        false);
    if (result != 0)
        return result;

    result = FSharpCompiler::compileProgram(
        inputType,
        fileName,
        Action::DumpMLIR,
        false);
    if (result != 0)
        return result;

    result = FSharpCompiler::compileProgram(
        inputType,
        fileName,
        Action::DumpMLIRTypeInference,
        false);
    if (result != 0)
        return result;

    result = FSharpCompiler::compileProgram(
        inputType,
        fileName,
        Action::DumpMLIRArith,
        false);
    if (result != 0)
        return result;

    result = FSharpCompiler::compileProgram(
        inputType,
        fileName,
        Action::DumpMLIRFunc,
        false);
    if (result != 0)
        return result;

    result = FSharpCompiler::compileProgram(
        inputType,
        fileName,
        Action::DumpMLIRBufferized,
        false);
    if (result != 0)
        return result;

    result = FSharpCompiler::compileProgram(
        inputType,
        fileName,
        Action::DumpMLIRLLVM,
        true);
    if (result != 0)
        return result;

    result = FSharpCompiler::compileProgram(
        inputType,
        fileName,
        Action::DumpLLVMIR,
        false);
    if (result != 0)
        return result;

    result = FSharpCompiler::compileProgram(
        inputType,
        fileName,
        Action::RunJIT,
        true);
    if (result != 0)
        return result;

    if (!emitExe)
        return result;

    result = FSharpCompiler::compileProgram(
        inputType,
        fileName,
        Action::EmitExecutable,
        true,
        std::move(executableOutputPath)
    );

    return result;
}

TEST(Assert, BasicAssertion)
{
    EXPECT_EQ(RunFullTestSuite(InputType::FSharp, "TestFiles/Assert.fs", false, "Assert"), 0);
}

TEST(HelloWorld, BasicAssertion)
{
    EXPECT_EQ(RunFullTestSuite(InputType::FSharp, "TestFiles/HelloWorld.fs", true, "HelloWorld"), 0);
}

TEST(HelloWorldVariable, BasicAssertion)
{
    EXPECT_EQ(RunFullTestSuite(InputType::FSharp, "TestFiles/HelloWorldVariable.fs", false, "HelloWorldVariable"), 0);
}

TEST(HelloWorldBranchConstant, BasicAssertion)
{
    EXPECT_EQ(RunFullTestSuite(InputType::FSharp, "TestFiles/HelloWorldBranchConstant.fs", false, "HelloWorldBranchConstant"), 0);
}

TEST(HelloWorldBranchNestedConstant, BasicAssertion)
{
    EXPECT_EQ(RunFullTestSuite(InputType::FSharp, "TestFiles/HelloWorldBranchNestedConstant.fs", false,
                     "HelloWorldBranchNestedConstant"), 0);
}

TEST(HelloWorldBranchRelation, BasicAssertion)
{
    EXPECT_EQ(RunFullTestSuite(InputType::FSharp, "TestFiles/HelloWorldBranchRelation.fs", false, "HelloWorldBranchRelation"), 0);
}

TEST(FunctionDefinition, BasicAssertion)
{
    EXPECT_EQ(RunFullTestSuite(InputType::FSharp, "TestFiles/FunctionDefinition.fs"), 0);
}

TEST(NestedFunctionDefinition, BasicAssertion)
{
    EXPECT_EQ(RunFullTestSuite(InputType::FSharp, "TestFiles/NestedFunctionDefinitions.fs"), 0);
}

TEST(SimpleAdd, BasicAssertion)
{
    EXPECT_EQ(RunFullTestSuite(InputType::FSharp, "TestFiles/SimpleAdd.fs", false, "SimpleAdd"), 0);
}

TEST(Logical, BasicAssertion)
{
    EXPECT_EQ(RunFullTestSuite(InputType::FSharp, "TestFiles/Logical.fs", false, "Logical"), 0);
}

TEST(LogicalNoDynamicReturn, BasicAssertion)
{
    EXPECT_EQ(RunFullTestSuite(InputType::FSharp, "TestFiles/LogicalNoDynamicReturn.fs", false, "LogicalNoDynamicReturn"), 0);
}

TEST(Relation, BasicAssertion)
{
    EXPECT_EQ(RunFullTestSuite(InputType::FSharp, "TestFiles/Relation.fs", false, "Relation"), 0);
}

TEST(Recursion, BasicAssertion)
{
    EXPECT_EQ(RunFullTestSuite(InputType::FSharp, "TestFiles/Recursion.fs", false, "Recursion"), 0);
}
