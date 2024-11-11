#ifndef FOO_OPS
#define FOO_OPS

include "mlir/IR/OpBase.td"

def Foo_Dialect : Dialect {
  let name = "foo";
  let cppNamespace = "::mlir::foo";
  let description = [{
    Lorem Ipsum
  }];
}

#endif // FOO_OPS
