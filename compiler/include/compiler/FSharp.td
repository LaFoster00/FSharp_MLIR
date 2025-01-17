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

//===----------------------------------------------------------------------===//
// FSharp Types
//===----------------------------------------------------------------------===//

def StringType : FSharp_Type<"String", "string"> {
  let summary = "A simple type for storing tensors filled with characters, but prettier printing.";
  let description = [{
    Strings represent a tensor<*xI8> with appropriate printing.
  }];

  /// Here we define the textual format of the type declaratively, which will
  /// automatically generate parser and printer logic. This will allow for
  /// instances of the type to be output as, for example:
  ///
  ///    !fsharp.string // a string.
  ///
  let assemblyFormat = "";
}

//===----------------------------------------------------------------------===//
// FSharp Attributes
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// FSharp Operations
//===----------------------------------------------------------------------===//

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
// Arithmetic Operations
//===----------------------------------------------------------------------===//
def FSharp_AddOp : FSharp_Op<"add"> {
  let summary = "addition operation";
  let description = [{
    This operation performs element-wise addition of two tensors.
  }];
  let arguments = (ins AnyTensor:$lhs, AnyTensor:$rhs);
  let results = (outs AnyTensor);
}

def FSharp_SubOp : FSharp_Op<"sub"> {
  let summary = "subtraction operation";
  let description = [{
    This operation performs element-wise subtraction of two tensors.
  }];
  let arguments = (ins AnyTensor:$lhs, AnyTensor:$rhs);
  let results = (outs AnyTensor);
}

def FSharp_MulOp : FSharp_Op<"mul"> {
  let summary = "multiplication operation";
  let description = [{
    This operation performs element-wise multiplication of two tensors.
  }];
  let arguments = (ins AnyTensor:$lhs, AnyTensor:$rhs);
  let results = (outs AnyTensor);
}

def FSharp_DivOp : FSharp_Op<"div"> {
  let summary = "division operation";
  let description = [{
    This operation performs element-wise division of two tensors.
  }];
  let arguments = (ins AnyTensor:$lhs, AnyTensor:$rhs);
  let results = (outs AnyTensor);
}

def FSharp_ModOp : FSharp_Op<"mod"> {
  let summary = "modulo operation";
  let description = [{
    This operation performs element-wise modulo of two tensors.
  }];
  let arguments = (ins AnyTensor:$lhs, AnyTensor:$rhs);
  let results = (outs AnyTensor);
}

#endif // FSHARP_OPS