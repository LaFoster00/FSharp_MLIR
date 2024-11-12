//
// Created by lasse on 11/11/24.
//

#ifndef FOO_H
#define FOO_H

#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/CallInterfaces.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#include "FooDialect.h.inc"

#define GET_OP_CLASSES
#include "FooOps.h.inc"

#endif //FOO_H
