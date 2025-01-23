//
// Created by lasse on 1/23/25.
//

#pragma once

#include "compiler/FSharpDialect.h"

namespace mlir::fsharp::utils{
/// Find a closure with the given name in the current scope or parent scopes.
mlir::fsharp::ClosureOp findClosureInScope(mlir::Operation *startOp, mlir::StringRef closureName);
}