//===- Ops.td - FSharp dialect operation definitions ----------*- tablegen -*-===//
//===----------------------------------------------------------------------===//
//
// Defines the operations of the FSharp dialect.
//
//===----------------------------------------------------------------------===//

#ifndef FSHARP_OPS
#define FSHARP_OPS

include "mlir/IR/OpBase.td"
include "mlir/Interfaces/FunctionInterfaces.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/AttrTypeBase.td"

// Provide a definition of the 'fsharp' dialect in the ODS framework so that we
// can define our operations.
def FSharp_Dialect : Dialect {
  let name = "fsharp";
  let cppNamespace = "::mlir::fsharp";
}

// Base class for fsharp dialect operations. This operation inherits from the base
// `Op` class in OpBase.td, and provides:
//   * The parent dialect of the operation.
//   * The mnemonic for the operation, or the name without the dialect prefix.
//   * A list of traits for the operation.
class FSharp_Op<string mnemonic, list<Trait> traits = []> :
    Op<FSharp_Dialect, mnemonic, traits>;

class FSharp_Type<string name, string typeMnemonic, list<Trait> traits = []>
  : TypeDef<FSharp_Dialect, name, traits> {
    let mnemonic = typeMnemonic;
  }

class FSharp_Attr<string name, string attrMnemonic, list<Trait> traits = []>
    : AttrDef<My_Dialect, name, traits> {
  let mnemonic = attrMnemonic;
  }

//===----------------------------------------------------------------------===//
// FSharp Types
//===----------------------------------------------------------------------===//

def StringType : FSharp_Type<"String", "string"> {
  let summary = "A simple type for storing tensors filled with characters, but prettier printing."
  let description = [{
    Strings represent a tensor<*xI8> with appropriate printing.
  }];

  let parameters = {ins }

  /// Here we define the textual format of the type declaratively, which will
  /// automatically generate parser and printer logic. This will allow for
  /// instances of the type to be output as, for example:
  ///
  ///    !fsharp.string "Bla" // a string containing the word bla.
  ///
  let assemblyFormat = " `\"` $data `\"`";
}

//===----------------------------------------------------------------------===//
// FSharp Attributes
//===----------------------------------------------------------------------===//

def StringAttr : FSharp_Attr<"String", "string"> {
  let summary = "A simple attribute for storing tensors filled with characters, but prettier printing."
  let description = [{
    Strings represent a tensor<*xI8> with appropriate printing.
  }];

  /// Here we've defined two parameters, one is a "self" type parameter, and the
  /// other is the integer value of the attribute. The self type parameter is
  /// specially handled by the assembly format.
  let parameters = (ins AttributeSelfTypeParameter<"">:$type, "std::string":$value);

  /// Here we've defined a custom builder for the type, that removes the need to pass
  /// in an MLIRContext instance; as it can be infered from the `type`.
  let builders = [
    AttrBuilderWithInferredContext<(ins "Type":$type,
                                        "const std::string &":$value), [{
      return $_get(type.getContext(), type, value);
    }]>
  ];

  /// Here we define the textual format of the type declaratively, which will
  /// automatically generate parser and printer logic. This will allow for
  /// instances of the type to be output as, for example:
  ///
  ///    !fsharp.string "Bla" // a string containing the word bla.
  ///
  let assemblyFormat = " `\"` $value `\"`";

  /// Indicate to the ODS generator that we do not want the default builders,
  /// as we have defined our own simpler ones.
  let skipDefaultBuilders = 1;
}


//===----------------------------------------------------------------------===//
// FSharp Operations
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// FuncOp
//===----------------------------------------------------------------------===//

def FuncOp : FSharp_Op<"func", [
    FunctionOpInterface, IsolatedFromAbove
  ]> {
  let summary = "user defined function operation";
  let description = [{
    The "fsharp.func" operation represents a user defined function. These are
    callable SSA-region operations that contain fsharp computations.

    Example:

    ```mlir
    fsharp.func @main() {
      %0 = fsharp.constant dense<5.500000e+00> : tensor<f64>
      %1 = fsharp.reshape(%0 : tensor<f64>) to tensor<2x2xf64>
      fsharp.print %1 : tensor<2x2xf64>
      fsharp.return
    }
    ```
  }];

  let arguments = (ins
    SymbolNameAttr:$sym_name,
    TypeAttrOf<FunctionType>:$function_type,
    OptionalAttr<DictArrayAttr>:$arg_attrs,
    OptionalAttr<DictArrayAttr>:$res_attrs
  );
  let regions = (region AnyRegion:$body);

  let builders = [OpBuilder<(ins
    "StringRef":$name, "FunctionType":$type,
    CArg<"ArrayRef<NamedAttribute>", "{}">:$attrs)
  >];

  let extraClassDeclaration = [{
    //===------------------------------------------------------------------===//
    // FunctionOpInterface Methods
    //===------------------------------------------------------------------===//

    /// Returns the argument types of this function.
    ArrayRef<Type> getArgumentTypes() { return getFunctionType().getInputs(); }

    /// Returns the result types of this function.
    ArrayRef<Type> getResultTypes() { return getFunctionType().getResults(); }

    Region *getCallableRegion() { return &getBody(); }
  }];

  let hasCustomAssemblyFormat = 1;
  let skipDefaultBuilders = 1;
}


//===----------------------------------------------------------------------===//
// GenericCallOp
//===----------------------------------------------------------------------===//

def GenericCallOp : FSharp_Op<"generic_call"> {
  let summary = "generic call operation";
  let description = [{
    Generic calls represent calls to a user defined function that needs to
    be specialized for the shape of its arguments. The callee name is attached
    as a symbol reference via an attribute. The arguments list must match the
    arguments expected by the callee. For example:

    ```mlir
     %4 = fsharp.generic_call @my_func(%1, %3)
           : (tensor<2x3xf64>, tensor<2x3xf64>) -> tensor<*xf64>
    ```

    This is only valid if a function named "my_func" exists and takes two
    arguments.
  }];

  // The generic call operation takes a symbol reference attribute as the
  // callee, and inputs for the call.
  let arguments = (ins FlatSymbolRefAttr:$callee, Variadic<F64Tensor>:$inputs);

  // The generic call operation returns a single value of TensorType.
  let results = (outs F64Tensor);

  // Specialize assembly printing and parsing using a declarative format.
  let assemblyFormat = [{
    $callee `(` $inputs `)` attr-dict `:` functional-type($inputs, results)
  }];

  // Add custom build methods for the generic call operation.
  let builders = [
    OpBuilder<(ins "StringRef":$callee, "ArrayRef<Value>":$arguments)>
  ];
}

//===----------------------------------------------------------------------===//
// PrintOp
//===----------------------------------------------------------------------===//

def PrintOp : FSharp_Op<"print"> {
  let summary = "print operation";
  let description = [{
    The "print" builtin operation prints a given input tensor, and produces
    no results.
  }];

  // The print operation takes an input tensor to print.
  let arguments = (ins F64Tensor:$input);

  let assemblyFormat = "$input attr-dict `:` type($input)";
}

#endif // FSHARP_OPS