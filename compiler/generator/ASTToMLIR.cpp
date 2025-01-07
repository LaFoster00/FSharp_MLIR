//
// Created by lasse on 1/7/25.
//
#include "ASTToMLIR.h"
#include "mlir/IR/Builders.h"
#include "Grammar.h"

mlir::ModuleOp generateMLIRFromAST(
    std::string_view source, mlir::MLIRContext& context)
{
    auto result = fsharpgrammar::Grammar::parse(source, true, false, true);
    mlir::OpBuilder builder(&context);

    // Create an MLIR module
    auto module = mlir::ModuleOp::create(builder.getUnknownLoc());

    // Convert AST to MLIR (implementation depends on your AST and IR design)
    // Example: traverseAST(root, module, builder);

    return module;
}
