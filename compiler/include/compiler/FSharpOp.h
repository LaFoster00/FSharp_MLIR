#pragma once

#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/FunctionInterfaces.h"
#include "mlir/IR/SymbolInterfaces.h"
#include "mlir/IR/SideEffectInterfaces.h"

namespace mlir {
    namespace fsharp {

#define GET_OP_CLASSES
#include "compiler/FSharp.h.inc"

    } // namespace fsharp
} // namespace mlir