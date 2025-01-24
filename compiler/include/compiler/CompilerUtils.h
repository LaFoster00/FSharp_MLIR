//
// Created by lasse on 1/23/25.
//

#pragma once

#include "compiler/FSharpDialect.h"

namespace mlir::fsharp::utils
{
    /// Find a closure with the given name in the current scope or parent scopes.
    mlir::fsharp::ClosureOp findClosureInScope(mlir::Operation* startOp, mlir::StringRef closureName);

    /// Find a function with the given name in the current scope or parent scopes.
    mlir::func::FuncOp findFunctionInScope(mlir::Operation* startOp, mlir::StringRef funcName);

    bool isImplicitTypeInferred(Operation *op);

    bool someOperandsInferred(Operation* op);

    /// A utility method that returns if the given operation has all of its
    /// operands inferred.
    bool allOperandsInferred(Operation* op);

    /// A utility method that returns if the given operation has a dynamically
    /// shaped result.
    bool returnsUnknownType(Operation* op);

    bool noOperandsInferred(Operation* op);

    bool returnsKnownType(Operation* op);
}
