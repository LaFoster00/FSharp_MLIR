//
// Created by lasse on 1/9/25.
//

#pragma once

#include "mlir/IR/Attributes.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"

namespace fsharpgrammar::ast
{
    struct Range;
}

namespace mlir
{
    class OpBuilder;
    class ModuleOp;
    class PatternRewriter;
    class Builder;
}

namespace fsharpgrammar::compiler
{
    /// Inserts printf so that we can call it from inside the application
      /// Create a function declaration for printf, the signature is:
      ///   * `i32 (i8*, ...)`
    mlir::LLVM::LLVMFunctionType getPrintfType(mlir::MLIRContext* context);

    /// Return a symbol reference to the printf function, inserting it into the
    /// module if necessary.
    mlir::FlatSymbolRefAttr getOrInsertPrintf(mlir::OpBuilder& builder,
                                                     mlir::ModuleOp module);

    /// Return a value representing an access into a global string with the given
    /// name, creating the string if necessary.
    mlir::Value getOrCreateGlobalString(mlir::Location loc, mlir::OpBuilder& builder,
                                               llvm::StringRef name, llvm::StringRef value,
                                               mlir::ModuleOp module);
}
