#pragma once

#include <memory>
#include <string>
#include <ast/ASTNode.h>
#include <llvm/ADT/StringRef.h>
#include <mlir/IR/BuiltinOps.h.inc>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/OwningOpRef.h>

enum class InputType { FSharp = 0, MLIR };

enum class Action
{
    None = 0,
    DumpST,
    DumpAST,
    DumpMLIR,
    DumpMLIRAffine,
    DumpMLIRLLVM,
    DumpLLVMIR,
    RunJIT
};

namespace fsharp::compiler
{
    class FSharpCompiler
    {
    public:
        static int compileProgram(InputType inputType, std::string inputFilename, Action emitAction,
                                  bool runOptimizations);

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
    };
}
