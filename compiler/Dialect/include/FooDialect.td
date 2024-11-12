#ifndef FOO_OPS
#define FOO_OPS

include "mlir/IR/OpBase.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

def Foo_Dialect : Dialect {
  let name = "Foo";
  let cppNamespace = "::Foo";
  let useDefaultTypePrinterParser = 1;
}

class Foo_Op<string mnemonic, list<Trait> traits = []> : Op<Foo_Dialect, mnemonic, traits>;

#endif // FOO_OPS
