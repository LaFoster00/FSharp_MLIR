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
include "mlir/IR/BuiltinAttributeInterfaces.td"
include "mlir/Dialect/Bufferization/IR/BufferizableOpInterface.td"
include "mlir/IR/CommonTypeConstraints.td"
include "mlir/IR/EnumAttr.td"
include "mlir/IR/OpAsmInterface.td"
include "mlir/Interfaces/CallInterfaces.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/InferTypeOpInterface.td"

// Provide a definition of the 'fsharp' dialect in the ODS framework so that we
// can define our operations.
def FSharp_Dialect : Dialect {
  let name = "fsharp";
  let cppNamespace = "::mlir::fsharp";
  let extraClassDeclaration = [{
            void registerTypes();
            void registerAttributes();
        }];
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

class FSharp_Attr<string name, string attrMnemonic, list<Trait> traits = [],
                                     string baseCppClass = "::mlir::Attribute">
    : AttrDef<FSharp_Dialect, name, traits, baseCppClass> {
  let mnemonic = ?;
  let attrName = "fsharp." # attrMnemonic;
  }

def TypeInferenceOpInterface : OpInterface<"TypeInference"> {
    let description = [{
        Interface to access a registered method to infer the return types for an
        operation that can be used during type inference.
    }];

    let methods = [
        InterfaceMethod<"Infer and set the output shape for the current operation. Returns 0 if not all of the types could be inferred, 1 if some could be inferred and 2 if all could be inferred..",
                        "int", "inferTypes">,
        InterfaceMethod<"Set all the unknown types to int.",
                        "void", "assumeTypes">
    ];
}

//===----------------------------------------------------------------------===//
// FSharp Operations
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// ClosureOp
//===----------------------------------------------------------------------===//

def ClosureOp : FSharp_Op<"closure", [FunctionOpInterface, IsolatedFromAbove, DeclareOpInterfaceMethods<TypeInferenceOpInterface>]> {
  let summary = "Represents a closure with implicit captures and explicit arguments.";
  let description = [{
    This operation represents a closure that captures variables from its
    enclosing scope (by value) at the time of closure creation. The closure
    also takes explicit arguments for parameters declared in its header.
    The body of the closure is stored in a single region.
  }];

  let arguments = (ins
    SymbolNameAttr:$sym_name,
    TypeAttrOf<FunctionType>:$function_type,
    OptionalAttr<DictArrayAttr>:$arg_attrs,
    OptionalAttr<DictArrayAttr>:$res_attrs);
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
// CallOp
//===----------------------------------------------------------------------===//

def CallOp : FSharp_Op<"call", [DeclareOpInterfaceMethods<TypeInferenceOpInterface>]> {
  let summary = "Call operation";
  let description = [{
    Calls represent calls to a user defined function that needs to
    be specialized for the shape of its arguments including implicit captures. The callee name is attached
    as a symbol reference via an attribute. The arguments list must match the
    arguments expected by the callee. For example:

    ```mlir
     %4 = fsharp.call @my_func(%1, %3)
           : (tensor<2x3xf64>, tensor<2x3xf64>) -> tensor<*xf64>
    ```

    This is only valid if a function named "my_func" exists and takes two
    arguments.
  }];

  // The generic call operation takes a symbol reference attribute as the
  // callee, and inputs for the call.
  let arguments = (ins FlatSymbolRefAttr:$callee, Variadic<AnyType>:$inputs);

  // The generic call operation returns a single value of TensorType.
  let results = (outs AnyType);

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

def PrintOp : FSharp_Op<"printf", [BufferizableOpInterface]> {
  let summary = "print operation";
  let description = [{
    The "print" builtin operation prints a given input tensor, and produces
    no results.
  }];

  // The print operation takes an input tensor to print.
  let arguments = (ins Variadic<AnyType>:$fmt_operands);

  let extraClassDeclaration = [{
    bool bufferizesToMemoryRead(OpOperand &opOperand,
                                const bufferization::AnalysisState &state)
    { return true; }

    bool bufferizesToMemoryWrite(OpOperand &opOperand,
                                const bufferization::AnalysisState &state)
    { return false; }

    bufferization::AliasingValueList getAliasingValues(OpOperand & opOperand, const bufferization::AnalysisState & state);
    LogicalResult bufferize(RewriterBase &rewriter, const bufferization::BufferizationOptions &options);

  }];


  let assemblyFormat = "`(` $fmt_operands `)` attr-dict `:` type($fmt_operands)";
  // Indicate that additional verification for this operation is necessary.
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// AddOp
//===----------------------------------------------------------------------===//

def AddOp : FSharp_Op<"add", [DeclareOpInterfaceMethods<TypeInferenceOpInterface>]> {
  let summary = "addition operation";
  let description = [{
    The "add" operation performs addition of any two types that are compatible.
  }];

  let arguments = (ins AnyType:$lhs, AnyType:$rhs);
  let results = (outs AnyType);

  let assemblyFormat = [{
    $lhs `,` $rhs attr-dict `:` functional-type(operands, results)
  }];

  // Allow building an AddOp with from the two input operands.
  let builders = [
    OpBuilder<(ins "Value":$lhs, "Value":$rhs)>
  ];
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// SubOp
//===----------------------------------------------------------------------===//

def SubOp : FSharp_Op<"sub", [DeclareOpInterfaceMethods<TypeInferenceOpInterface>]> {
  let summary = "subtraction operation";
  let description = [{
    The "sub" operation performs subtraction of any two types that are compatible.
  }];

  let arguments = (ins AnyType:$lhs, AnyType:$rhs);
  let results = (outs AnyType);

  let assemblyFormat = [{
    $lhs `,` $rhs attr-dict `:` functional-type(operands, results)
  }];

  // Allow building an AddOp with from the two input operands.
  let builders = [
    OpBuilder<(ins "Value":$lhs, "Value":$rhs)>
  ];
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// MulOp
//===----------------------------------------------------------------------===//

def MulOp : FSharp_Op<"mul", [DeclareOpInterfaceMethods<TypeInferenceOpInterface>]> {
  let summary = "multiplication operation";
  let description = [{
    The "sub" operation performs multiplication of any two types that are compatible.
  }];

  let arguments = (ins AnyType:$lhs, AnyType:$rhs);
  let results = (outs AnyType);

  let assemblyFormat = [{
    $lhs `,` $rhs attr-dict `:` functional-type(operands, results)
  }];

  // Allow building an AddOp with from the two input operands.
  let builders = [
    OpBuilder<(ins "Value":$lhs, "Value":$rhs)>
  ];
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// DivOp
//===----------------------------------------------------------------------===//

def DivOp : FSharp_Op<"div", [DeclareOpInterfaceMethods<TypeInferenceOpInterface>]> {
  let summary = "division operation";
  let description = [{
    The "sub" operation performs division of any two types that are compatible.
  }];

  let arguments = (ins AnyType:$lhs, AnyType:$rhs);
  let results = (outs AnyType);

  let assemblyFormat = [{
    $lhs `,` $rhs attr-dict `:` functional-type(operands, results)
  }];

  // Allow building an AddOp with from the two input operands.
  let builders = [
    OpBuilder<(ins "Value":$lhs, "Value":$rhs)>
  ];
  let hasVerifier = 1;
}

def FSharp_ModOp : FSharp_Op<"mod", [DeclareOpInterfaceMethods<TypeInferenceOpInterface>]> {
  let summary = "modulo operation";
  let description = [{
    This operation performs element-wise modulo of two tensors.
  }];

  let arguments = (ins AnyType:$lhs, AnyType:$rhs);
  let results = (outs AnyType);

  let assemblyFormat = [{
    $lhs `,` $rhs attr-dict `:` functional-type(operands, results)
  }];

  // Allow building an AddOp with from the two input operands.
  let builders = [
    OpBuilder<(ins "Value":$lhs, "Value":$rhs)>
  ];
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// LogicalOps
//===----------------------------------------------------------------------===//
def FSharp_AndOp : FSharp_Op<"and", [DeclareOpInterfaceMethods<TypeInferenceOpInterface>]> {
  let summary = "logical and operation";
  let description = [{
    This operation performs element-wise logical and of two tensors.
  }];

  let arguments = (ins AnyType:$lhs, AnyType:$rhs);
  let results = (outs AnyType);

  let assemblyFormat = [{
    $lhs `,` $rhs attr-dict `:` functional-type(operands, results)
  }];

  // Allow building an AddOp with from the two input operands.
  let builders = [
    OpBuilder<(ins "Value":$lhs, "Value":$rhs)>
  ];
  let hasVerifier = 1;
}

def FSharp_OrOp : FSharp_Op<"or", [DeclareOpInterfaceMethods<TypeInferenceOpInterface>]> {
  let summary = "logical or operation";
  let description = [{
    This operation performs element-wise logical or of two tensors.
  }];

  let arguments = (ins AnyType:$lhs, AnyType:$rhs);
  let results = (outs AnyType);

  let assemblyFormat = [{
    $lhs `,` $rhs attr-dict `:` functional-type(operands, results)
  }];

  // Allow building an AddOp with from the two input operands.
  let builders = [
    OpBuilder<(ins "Value":$lhs, "Value":$rhs)>
  ];
  let hasVerifier = 1;
}

#endif // FSHARP_OPS