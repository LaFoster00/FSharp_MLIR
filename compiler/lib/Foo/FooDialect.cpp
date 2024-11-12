//
// Created by lasse on 11/11/24.
//

#include "mlir/IR/Dialect.h"

#include "Foo/FooDialect.h"
#include "Foo/FooOps.h"

using namespace mlir;
using namespace Foo;

#include "Foo/FooDialect.cpp.inc"

void FooDialect::initialize()
{
    addOperations<
#define GET_TYPEDEF_LIST
#include "Foo/FooTypes.h.inc"
        >();
    addOperations<
#define GET_OP_LIST
#include "Foo/Foo.cpp.inc"
        >();
}