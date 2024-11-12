//
// Created by lasse on 11/11/24.
//

#include "include/FooDialect.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/Target/LLVMIR/ModuleImport.h"

int main(int argc, char **argv) {
  mlir::MLIRContext context;
  context.getOrLoadDialect<Foo::FooDialect>();
  return 0;
}
