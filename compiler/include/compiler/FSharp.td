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

def FSharp_StringAttr : FSharp_Attr<"FSharp_String", "fsharp_string", [ElementsAttrInterface],
        "DenseElementsAttr"> {
  let summary = "An Attribute containing a string";
  let description = [{
    Syntax:

    ```
    string-attribute ::= string-literal (`:` type)?
    ```

    A string attribute is an attribute that represents a string literal value.

    Examples:

    ```mlir
    "An important string"
    "string with a type" : !dialect.string
    ```
  }];
  let parameters = (ins AttributeSelfTypeParameter<"", "ShapedType">:$type,
                        StringRefParameter<"">:$value);
  let builders = [
    AttrBuilderWithInferredContext<(ins "const Twine &":$bytes, "Type":$type)>,
  ];

    let extraClassDeclaration = [{
        using DenseElementsAttr::empty;
        using DenseElementsAttr::getNumElements;
        using DenseElementsAttr::getElementType;
        using DenseElementsAttr::getValues;
        using DenseElementsAttr::isSplat;
        using DenseElementsAttr::size;
        using DenseElementsAttr::value_begin;

        /// The set of data types that can be iterated by this attribute.
        using ContiguousIterableTypesT = std::tuple<
            // Char types.
            char8_t, char16_t, char32_t
        >;
        using NonContiguousIterableTypesT = std::tuple<
            Attribute,
            // Char types.
            char8_t, char16_t, char32_t
        >;

        /// Provide a `try_value_begin_impl` to enable iteration within
        /// ElementsAttr.
        template <typename T>
        auto try_value_begin_impl(OverloadToken<T>) const {
          return try_value_begin<T>();
        }

        protected:
            friend DenseElementsAttr;

        public:
        using ValueType = StringRef;

        /// If the value of this string is prefixed with a dialect namespace,
        /// returns the dialect corresponding to that namespace if it is loaded,
        /// nullptr otherwise. For example, the string `llvm.fastmathflags` would
        /// return the LLVM dialect, assuming it is loaded in the context.
        Dialect *getReferencedDialect() const;

        /// Enable conversion to StringRef.
        operator StringRef() const { return getValue(); }

        /// Returns the underlying string value
        StringRef strref() const { return getValue(); }

        /// Convert the underling value to an std::string.
        std::string str() const { return getValue().str(); }

        /// Return a pointer to the start of the string data.
        const char *data() const { return getValue().data(); }

        /// Return the number of bytes in this string.
        size_t size() const { return getValue().size(); }

        /// Return whether the string is empty.
        bool empty() const { return getValue().empty(); }

        /// Iterate over the underlying string data.
        StringRef::iterator begin() const { return getValue().begin(); }
        StringRef::iterator end() const { return getValue().end(); }

        /// Compare the underlying string value to the one in `rhs`.
        int compare(FSharp_StringAttr rhs) const {
          if (*this == rhs)
            return 0;
          return getValue().compare(rhs.getValue());
        }

        private:
        friend MLIRContext;
        public:
        }];
    let genStorageClass = 1;
    let skipDefaultBuilders = 1;
}



//===----------------------------------------------------------------------===//
// FSharp Operations
//===----------------------------------------------------------------------===//

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
  let arguments = (ins Variadic<AnyType>:$operands);


  let assemblyFormat = "`(` $operands `)` attr-dict`:` type($operands)";
}

#endif // FSHARP_OPS

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