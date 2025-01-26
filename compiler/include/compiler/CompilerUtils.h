//
// Created by lasse on 1/23/25.
//

#pragma once

#include "compiler/FSharpDialect.h"

#define GENERATE_OP_CONVERSION_PATTERN(op_name) \
struct op_name##Lowering : public OpConversionPattern<fsharp::op_name> \
{ \
using OpConversionPattern<fsharp::op_name>::OpConversionPattern; \
LogicalResult matchAndRewrite(fsharp::op_name op, OpAdaptor adaptor, ConversionPatternRewriter& rewriter) const final \
{

#define END_GENERATE_OP_CONVERSION_PATTERN() }};

namespace mlir::fsharp::utils
{
    /// Find a closure with the given name in the current scope or parent scopes.
    mlir::fsharp::ClosureOp findClosureInScope(mlir::Operation* startOp, mlir::StringRef closureName);

    /// Find a function with the given name in the current scope or parent scopes.
    mlir::func::FuncOp findFunctionInScope(mlir::Operation* startOp, mlir::StringRef funcName);

    bool isImplicitTypeInferred(Operation* op);

    bool someOperandsInferred(Operation* op);

    /// A utility method that returns if the given operation has all of its
    /// operands inferred.
    bool allOperandsInferred(Operation* op);

    /// A utility method that returns if the given operation has a dynamically
    /// shaped result.
    bool returnsUnknownType(Operation* op);

    bool noOperandsInferred(Operation* op);

    bool returnsKnownType(Operation* op);

    void addOrUpdateAttrDictEntry(mlir::Operation* op, mlir::StringRef key, mlir::Attribute value);

    void deleteAttrDictEntry(mlir::Operation* op, mlir::StringRef key);

    // Helper function to create a global memref for the string.
    Value createGlobalMemrefForString(Location loc, StringRef stringValue,
                                      OpBuilder& builder, ModuleOp module, Operation* op);

    llvm::SmallVector<std::tuple<char, mlir::Type>, 4> getFormatSpecifiedTypes(llvm::StringRef format, mlir::MLIRContext* context);
}
