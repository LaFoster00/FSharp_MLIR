//
// Created by lasse on 11/12/24.
//

#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/TypeSwitch.h"

#include "Foo/FooDialect.h"
#include "Foo/FooTypes.h"

#define GET_TYPEDEF_CLASSES
#include "Foo/FooTypes.cpp.inc"
#undef GET_TYPEDEF_CLASSES