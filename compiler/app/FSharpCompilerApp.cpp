//
// Created by lasse on 1/27/25.
//

#include <compiler/ASTToMLIR.h>
#include <compiler/Compiler.h>

#include <llvm/Support/CommandLine.h>

namespace cl = llvm::cl;

// Define a category for all the compiler specific operations so that we can group them better in the help message.
cl::OptionCategory CompilerCategory("FSharp Compiler Options", "Specific options for FSharp compiler");


// The input file to read
static cl::opt<std::string> inputFilename(cl::Positional,
                                          cl::desc("<input fsharp file>"),
                                          cl::init("-"),
                                          cl::value_desc("filename"),
                                          cl::cat(CompilerCategory));

// The optional executable output file
static cl::opt<std::string> executableOutputPath("o", cl::desc("Specify the output filename for the executable"),
                                                 cl::init("-"),
                                                 cl::value_desc("filename"),
                                                 cl::cat(CompilerCategory));

// The type of the input file
static cl::opt<enum InputType> inputType(
    "x", cl::init(InputType::FSharp),
    cl::desc("Decided the kind of output desired"),
    cl::cat(CompilerCategory),
    cl::values(clEnumValN(InputType::FSharp, "fsharp",
                          "load the input file as a FSharp source.")),
    cl::values(clEnumValN(InputType::MLIR, "mlir",
                          "load the input file as an MLIR file")));

// The action the compiler should perfom
static cl::opt<enum Action> emitAction(
    "emit", cl::desc("Select the kind of output desired"),
    cl::cat(CompilerCategory),
    cl::values(clEnumValN(Action::DumpST, "st",
                          "output the ST dump")),
    cl::values(clEnumValN(Action::DumpAST, "ast",
                          "output the AST dump")),
    cl::values(clEnumValN(Action::DumpMLIR, "mlir",
                          "output the MLIR dump")),
    cl::values(clEnumValN(Action::DumpMLIRTypeInference, "mlir-typed",
                          "output the MLIR dump after type inference")),
    cl::values(clEnumValN(Action::DumpMLIRArith, "mlir-arith",
                          "output the MLIR dump after arith lowering")),
    cl::values(clEnumValN(Action::DumpMLIRFunc, "mlir-func",
                          "output the MLIR dump after func lowering")),
    cl::values(clEnumValN(Action::DumpMLIRBufferized, "mlir-buff",
                          "output the MLIR dump after bufferization")),
    cl::values(clEnumValN(Action::DumpMLIRLLVM, "mlir-llvm",
                          "output the MLIR dump after llvm lowering")),
    cl::values(clEnumValN(Action::DumpLLVMIR, "llvm",
                          "output the LLVM IR dump")),
    cl::values(clEnumValN(Action::RunJIT, "jit",
                          "JIT the code and run it by executing the input script")),
    cl::values(clEnumValN(Action::EmitExecutable, "exe",
                          "Emit an executable from the input script")));

// Enable optimizations
static cl::opt<bool> enableOpt("opt",
                               cl::desc("Enable optimizations"),
                               cl::cat(CompilerCategory));

int main(int argc, char** argv)
{
    cl::HideUnrelatedOptions(CompilerCategory);
    cl::ParseCommandLineOptions(argc, argv, "fsharp compiler\n");

    fsharp::compiler::FSharpCompiler::compileProgram(inputType, inputFilename, emitAction, enableOpt,
                                                     executableOutputPath);
}
