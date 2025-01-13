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