//
// Created by lasse on 1/7/25.
//
#pragma once


#include <memory>

namespace fsharpgrammar::ast
{
    class Main;
}

namespace mlir {
    class MLIRContext;
    template <typename OpTy>
    class OwningOpRef;
    class ModuleOp;
} // namespace mlir

namespace fsharp::compiler
{
    class MLIRGen
    {
    public:
        /// Emit IR for the given Toy moduleAST, returns a newly created MLIR module
        /// or nullptr on failure.
        static mlir::OwningOpRef<mlir::ModuleOp> mlirGen(mlir::MLIRContext &context, std::string_view source, std::string_view source_filename = "File");
        static mlir::OwningOpRef<mlir::ModuleOp> mlirGen(mlir::MLIRContext& context, std::unique_ptr<fsharpgrammar::ast::Main>& ast,std::string_view source_filename = "File");
    };
}
