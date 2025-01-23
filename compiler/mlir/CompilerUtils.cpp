//
// Created by lasse on 1/23/25.
//
#include "compiler/CompilerUtils.h"

namespace mlir::fsharp::utils{
mlir::fsharp::ClosureOp findClosureInScope(mlir::Operation *startOp, mlir::StringRef closureName)
{
    mlir::Operation *currentOp = startOp;

    // Traverse up through parent operations (or regions) to find the closure
    while (currentOp) {
        // Check if the current operation has a SymbolTable
        if (currentOp->hasTrait<mlir::OpTrait::SymbolTable>()) {
            // Try to lookup the closure in the current SymbolTable
            mlir::Operation *closure = mlir::SymbolTable::lookupSymbolIn(currentOp, closureName);
            if (auto closure_op = mlir::dyn_cast<mlir::fsharp::ClosureOp>(closure)) {
                return closure_op; // Found the closure
            }
        }

        // Move to the parent operation
        currentOp = currentOp->getParentOp();
    }

    // If no closure was found, return nullptr
    return nullptr;
}
}