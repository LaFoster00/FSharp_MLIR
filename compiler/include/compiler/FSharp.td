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
        InterfaceMethod<"Infer the output shape for the current operation based on the operands which are all typed. ",
                        "void", "inferFromOperands">,
        InterfaceMethod<"Infer the operand types from the return type of the operation.",
                        "void", "inferFromReturnType">,
        InterfaceMethod<"Set all the unknown types to int.",
                        "void", "inferFromUnknown">
    ];
}

def ImplicitTypeInferenceOpInterface : OpInterface<"ImplicitTypeInference"> {
    let description = [{
        Interface to mark an op as implicitly inferring its type from its operands, without any effect on the evaluated
        type.
    }];
}

//===----------------------------------------------------------------------===//
// FSharp Operations
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// ConstantOp
//===----------------------------------------------------------------------===//

// We define a fsharp operation by inheriting from our base 'FSharp_Op' class above.
// Here we provide the mnemonic and a list of traits for the operation. The
// constant operation is marked as 'Pure' as it is a pure operation
// and may be removed if dead.
def ConstantOp : FSharp_Op<"constant", [Pure, AllTypesMatch<["value", "result"]>]> {
  // Provide a summary and description for this operation. This can be used to
  // auto-generate documentation of the operations within our dialect.
  let summary = "constant";
  let description = [{
    Constant operation turns a literal into an SSA value. The data is attached
    to the operation as an attribute. For example:

    ```mlir
      %0 = fsharp.constant dense<[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]>
                        : tensor<2x3xf64>
    ```
  }];

  // The constant operation takes an attribute as the only input.
  let arguments = (ins TypedAttrInterface:$value);

  // The constant operation returns a single value of anytype.
  let results = (outs AnyType:$result);

  let extraClassDeclaration = [{
      /// Whether the constant op can be constructed with a particular value and
      /// type.
      static bool isBuildableWith(Attribute value, Type type){
        return mlir::arith::ConstantOp::isBuildableWith(value, type);
      }
    }];

  // Indicate that the operation has a custom parser and printer method.
  let assemblyFormat = "attr-dict $value";
}

//===----------------------------------------------------------------------===//
// ClosureOp
//===----------------------------------------------------------------------===//

def ClosureOp : FSharp_Op<"closure",
  [Symbol, SymbolTable, FunctionOpInterface, DeclareOpInterfaceMethods<TypeInferenceOpInterface>]> {
  let summary = "Represents a closure with implicit captures and explicit arguments.";
  let description = [{
    This operation represents a closure that captures variables from its
    enclosing scope (by value) at the time of closure creation. The closure
    also takes explicit arguments for parameters declared in its header.
    The body of the closure is stored in a single region.
  }];

  let arguments = (ins
    SymbolNameAttr:$sym_name,  // Name of the closure
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

      // Update the function signature of this closure so that it matches the operation types inside the body.
      void updateSignatureFromBody();
    }];

  let hasCustomAssemblyFormat = 1;
  let skipDefaultBuilders = 1;
}


//===----------------------------------------------------------------------===//
// CallOp
//===----------------------------------------------------------------------===//

def CallOp : FSharp_Op<"call", [CallOpInterface, DeclareOpInterfaceMethods<TypeInferenceOpInterface>, HasParent<"ClosureOp">]> {
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
  let arguments = (ins FlatSymbolRefAttr:$callee, Variadic<AnyType>:$operands);

  // The generic call operation returns a single value of TensorType.
  let results = (outs Optional<AnyType>);

  // Add custom build methods for the generic call operation.
  let builders = [
        OpBuilder<(ins "ClosureOp":$callee, CArg<"ValueRange", "{}">:$operands), [{
          $_state.addOperands(operands);
          $_state.addAttribute("callee", SymbolRefAttr::get(callee));
          $_state.addTypes(callee.getFunctionType().getResults());
        }]>,
        OpBuilder<(ins "SymbolRefAttr":$callee, "TypeRange":$results,
          CArg<"ValueRange", "{}">:$operands), [{
          $_state.addOperands(operands);
          $_state.addAttribute("callee", callee);
          $_state.addTypes(results);
        }]>,
        OpBuilder<(ins "StringAttr":$callee, "TypeRange":$results,
          CArg<"ValueRange", "{}">:$operands), [{
          build($_builder, $_state, SymbolRefAttr::get(callee), results, operands);
        }]>,
        OpBuilder<(ins "StringRef":$callee, "TypeRange":$results,
          CArg<"ValueRange", "{}">:$operands), [{
          build($_builder, $_state, StringAttr::get($_builder.getContext(), callee),
                results, operands);
        }]>
    ];

    let extraClassDeclaration = [{
        FunctionType getCalleeType();

        /// Get the argument operands to the called function.
        operand_range getArgOperands() {
          return {arg_operand_begin(), arg_operand_end()};
        }

        MutableOperandRange getArgOperandsMutable() {
          return getOperandsMutable();
        }

        operand_iterator arg_operand_begin() { return operand_begin(); }
        operand_iterator arg_operand_end() { return operand_end(); }

        /// Return the callee of this operation.
        CallInterfaceCallable getCallableForCallee() {
          return (*this)->getAttrOfType<SymbolRefAttr>("callee");
        }

        /// Set the callee for this operation.
        void setCalleeFromCallable(CallInterfaceCallable callee) {
          (*this)->setAttr("callee", callee.get<SymbolRefAttr>());
        }
      }];

    let assemblyFormat = [{
        $callee `(` $operands `)` attr-dict `:` functional-type($operands, results)
    }];
}

//===----------------------------------------------------------------------===//
// ReturnOp
//===----------------------------------------------------------------------===//

def ReturnOp : FSharp_Op<"return", [Pure, HasParent<"ClosureOp">, DeclareOpInterfaceMethods<TypeInferenceOpInterface>, Terminator]> {
  let summary = "Function return operation";
  let description = [{
    The `FSharp.return` operation represents a return operation within a function.
    The operation takes variable number of operands and produces no results.
    The operand number and types must match the signature of the function
    that contains the operation.

    Example:

    ```mlir
    fsharp.closure @foo() : (i32, f8) {
      ...
      return %0, %1 : i32, f8
    }
    ```
  }];

  let arguments = (ins Optional<AnyType>:$operand);

  let builders = [OpBuilder<(ins), [{
    build($_builder, $_state, nullptr);
  }]>];

  let assemblyFormat = "attr-dict ($operand^ `:` type($operand))?";
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// PrintOp
//===----------------------------------------------------------------------===//

def PrintOp : FSharp_Op<"printf", [BufferizableOpInterface, DeclareOpInterfaceMethods<ImplicitTypeInferenceOpInterface>]> {
  let summary = "print operation";
  let description = [{
    The "print" builtin operation prints a given input tensor, and produces
    no results.
  }];

  // The print operation takes an input tensor to print.
  let arguments = (ins StrAttr:$fmt_string, Variadic<AnyType>:$fmt_operands);

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


  let assemblyFormat = "$fmt_string ` ` `(` $fmt_operands `)` attr-dict `:` type($fmt_operands)";
  // Indicate that additional verification for this operation is necessary.
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// AssertOp
//===----------------------------------------------------------------------===//

def AssertOp : FSharp_Op<"assert", [DeclareOpInterfaceMethods<ImplicitTypeInferenceOpInterface>]> {
  let summary = "Assert operation";
  let description = [{
    The assert operation verifies that the condition evaluates to true.
    If the condition is false, an assertion failure occurs with an optional message.

    Example:
    ```mlir
      %1 = fsharp.const true : i1
      fsharp.assert(%1) ["message" = "Condition failed"]
    ```
  }];

  // The operation takes a single operand of AnyType (expected to be i1).
  let arguments = (ins AnyType:$condition, StrAttr:$message);

  // No results as assert doesn't return anything.
  let results = (outs);

  // Custom build methods for the assert operation.
  let builders = [
    OpBuilder<(ins "Value":$condition), [{
      $_state.addOperands(condition);
      // Set a default message if none is provided.
      StringRef defaultMessage = "Assertion failed";
      $_state.addAttribute("message", StringAttr::get($_builder.getContext(), defaultMessage));
    }]>,
    OpBuilder<(ins "Value":$condition, "StringAttr":$msg), [{
      $_state.addOperands(condition);
      $_state.addAttribute("message", msg);
    }]>,

  ];

  let skipDefaultBuilders = 1;
  let hasVerifier = 1;

  let assemblyFormat = [{
    $condition attr-dict `:` type($condition)
  }];
}

//===----------------------------------------------------------------------===//
// AddOp
//===----------------------------------------------------------------------===//

def AddOp : FSharp_Op<"add", [DeclareOpInterfaceMethods<TypeInferenceOpInterface>, HasParent<"ClosureOp">]> {
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

def SubOp : FSharp_Op<"sub", [DeclareOpInterfaceMethods<TypeInferenceOpInterface>, HasParent<"ClosureOp">]> {
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

def MulOp : FSharp_Op<"mul", [DeclareOpInterfaceMethods<TypeInferenceOpInterface>, HasParent<"ClosureOp">]> {
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

def DivOp : FSharp_Op<"div", [DeclareOpInterfaceMethods<TypeInferenceOpInterface>, HasParent<"ClosureOp">]> {
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

//===----------------------------------------------------------------------===//
// ModOp
//===----------------------------------------------------------------------===//

def FSharp_ModOp : FSharp_Op<"mod", [DeclareOpInterfaceMethods<TypeInferenceOpInterface>, HasParent<"ClosureOp">]> {
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
//===----------------------------------------------------------------------===//
// AndOp
//===----------------------------------------------------------------------===//
def FSharp_AndOp : FSharp_Op<"and", [DeclareOpInterfaceMethods<TypeInferenceOpInterface>]> {
  let summary = "and operation";
  let description = [{
    This operation performs element-wise adn of two tensors.
  }];

  let arguments = (ins AnyType:$lhs, AnyType:$rhs);
  let results = (outs AnyType);

  let assemblyFormat = [{
    $lhs `,` $rhs attr-dict `:` functional-type(operands, results)
  }];

  // Allow building an AndOp with from the two input operands.
  let builders = [
    OpBuilder<(ins "Value":$lhs, "Value":$rhs)>
  ];
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// OrOp
//===----------------------------------------------------------------------===//
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

//===----------------------------------------------------------------------===//
// EqualityOps
//===----------------------------------------------------------------------===//
//===----------------------------------------------------------------------===//
// EqualOp
//===----------------------------------------------------------------------===//
def FSharp_EqualOp : FSharp_Op<"equal", [DeclareOpInterfaceMethods<TypeInferenceOpInterface>]> {
  let summary = "equality equal operation";
  let description = [{
    This operation performs element-wise equality equal of two tensors.
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
// NotEqualOp
//===----------------------------------------------------------------------===//
def FSharp_NotEqualOp : FSharp_Op<"notequal", [DeclareOpInterfaceMethods<TypeInferenceOpInterface>]> {
  let summary = "equality not equal operation";
  let description = [{
    This operation performs element-wise equality not equal of two tensors.
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
// RelationOps
//===----------------------------------------------------------------------===//
//===----------------------------------------------------------------------===//
// LessOp
//===----------------------------------------------------------------------===//
def FSharp_LessOp : FSharp_Op<"less", [DeclareOpInterfaceMethods<TypeInferenceOpInterface>]> {
  let summary = "less than operation";
  let description = [{
    This operation performs element-wise less than of two tensors.
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
// LessEqualOp
//===----------------------------------------------------------------------===//
def FSharp_LessEqualOp : FSharp_Op<"lessequal", [DeclareOpInterfaceMethods<TypeInferenceOpInterface>]> {
  let summary = "less than or equal operation";
  let description = [{
    This operation performs element-wise less than or equal of two tensors.
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
// GreaterOp
//===----------------------------------------------------------------------===//
def FSharp_GreaterOp : FSharp_Op<"greater", [DeclareOpInterfaceMethods<TypeInferenceOpInterface>]> {
  let summary = "greater than operation";
  let description = [{
    This operation performs element-wise greater than of two tensors.
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
// GreaterEqualOp
//===----------------------------------------------------------------------===//
def FSharp_GreaterEqualOp : FSharp_Op<"greaterequal", [DeclareOpInterfaceMethods<TypeInferenceOpInterface>]> {
  let summary = "greater than or equal operation";
  let description = [{
    This operation performs element-wise greater than or equal of two tensors.
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