//
// Created by lasse on 1/9/25.
//
#pragma once

#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/CallInterfaces.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

/// Include the auto-generated header file containing the declaration of the fsharp
/// dialect.
#include "compiler/Dialect.h.inc"

/// Include the auto-generated header file containing the declarations of the
/// fsharp operations.
#define GET_OP_CLASSES
#include "compiler/Ops.h.inc"
