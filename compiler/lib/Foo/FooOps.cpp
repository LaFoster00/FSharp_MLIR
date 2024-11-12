//
// Created by lasse on 11/12/24.
//

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Bytecode/BytecodeOpInterface.h"

#include "Foo/FooOps.h"

#include "Foo/FooDialect.h"
#include "Foo/FooTypes.h"

#define GET_OP_CLASSES
#include "Foo/Foo.cpp.inc"