#pragma once


#include "ast/ASTNode.h"

#include <llvm/ADT/StringRef.h>

enum class InputType { FSharp = 0, MLIR };

enum class Action
{
    None = 0,
    DumpST,
    DumpAST,
    DumpMLIR,
    DumpMLIRTypeInference,
    DumpMLIRArith,
    DumpMLIRFunc,
    DumpMLIRBufferized,
    DumpMLIRLLVM,
    DumpLLVMIR,
    RunJIT,
    EmitExecutable
};

namespace fsharp::compiler
{
    class FSharpCompiler
    {
    public:
        static int compileProgram(InputType inputType, std::string_view inputFilename, Action emitAction,
                                  bool runOptimizations, std::optional<std::string> executableOutputPath = std::nullopt);

    private:
        static inline InputType inputType = InputType::FSharp;
        static inline std::string inputFilename = "-";
        static inline Action emitAction;
        static inline bool runOptimizations = false;

        /// Returns a FSharp AST resulting from parsing the file or a nullptr on error.
        static std::unique_ptr<fsharpgrammar::ast::Main> parseInputFile(llvm::StringRef filename);
        static int loadMLIR(mlir::MLIRContext& context,
                            mlir::OwningOpRef<mlir::ModuleOp>& module);
        static int loadAndProcessMLIR(mlir::MLIRContext& context,
                                      mlir::OwningOpRef<mlir::ModuleOp>& module);
        static int dumpST();
        static int dumpAST();
        static int dumpLLVMIR(mlir::ModuleOp module);
        static int runJit(mlir::ModuleOp module);
        static int emitExecutable(mlir::ModuleOp module, const std::string& outputFilePath, bool writeNewDbgInfoFormat=true);
    };
}
