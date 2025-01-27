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

namespace fsharpgrammar::ast
{
    struct Range;
    class IASTNode;
    struct INodeAlternative;
    class Type;
}

namespace mlir::fsharp::utils
{
    // Get the string representation of a type.
    std::string getTypeString(mlir::Type type);

    mlir::Location loc(const fsharpgrammar::ast::Range& range,
                       std::string_view filename,
                       mlir::MLIRContext* context);

    mlir::Location loc(const fsharpgrammar::ast::IASTNode& node,
                       std::string_view filename,
                       mlir::MLIRContext* context);

    mlir::Location loc(const fsharpgrammar::ast::INodeAlternative& node_alternative,
                       std::string_view filename,
                       mlir::MLIRContext* context);

    // Get the mlir type for a given type name or ast type node
    mlir::Type getMLIRType(const std::string_view type_name, mlir::MLIRContext* context);
    mlir::Type getMLIRType(const fsharpgrammar::ast::Type& type, mlir::MLIRContext* context, mlir::Location loc);

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

    llvm::SmallVector<std::tuple<char, mlir::Type>, 4> getFormatSpecifiedTypes(
        llvm::StringRef format, mlir::MLIRContext* context);
}
