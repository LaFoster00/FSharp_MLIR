//
// Created by lasse on 11/11/24.
//

#include "../../../extern/llvm-project/mlir/examples/toy/Ch4/include/toy/MLIRGen.h"
#include "include/FooDialect.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/Target/LLVMIR/ModuleImport.h"

int main(int argc, char **argv) {
  mlir::MLIRContext context;
  context.getOrLoadDialect<mlir::foo::FooDialect>();
  return 0;
}
