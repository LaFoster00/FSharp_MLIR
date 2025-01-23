//
// Created by lasse on 10/01/2025.
//

#include "compiler/FSharpDialect.h"

#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/Value.h"
#include "mlir/Interfaces/FunctionImplementation.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Casting.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"

#include <algorithm>
#include <ranges>
#include <string>
#include <fmt/format.h>
#include <llvm/ADT/MapVector.h>
#include <mlir/IR/IRMapping.h>

namespace fsharpgrammar::ast
{
    class Type;
}

using namespace mlir;
using namespace mlir::fsharp;

#include "compiler/FSharpInterfacesDefs.cpp.inc"

#include "compiler/FSharpDialect.cpp.inc"

//===----------------------------------------------------------------------===//
// FSharpDialect
//===----------------------------------------------------------------------===//

/// Dialect initialization, the instance will be owned by the context. This is
/// the point of registration of types and operations for the dialect.
void FSharpDialect::initialize()
{
    addOperations<
#define GET_OP_LIST
#include "compiler/FSharp.cpp.inc"
    >();
    registerTypes();
    registerAttributes();
}

//===----------------------------------------------------------------------===//
// FSharp Types
//===----------------------------------------------------------------------===//

#define GET_TYPEDEF_CLASSES
#include "compiler/FSharpTypes.cpp.inc"

void FSharpDialect::registerTypes()
{
    addTypes<
#define GET_TYPEDEF_LIST
#include "compiler/FSharpTypes.cpp.inc"
    >();
}

//===----------------------------------------------------------------------===//
// FSharp Attributes
//===----------------------------------------------------------------------===//

#define GET_ATTRDEF_CLASSES
#include "compiler/FSharpAttrDefs.cpp.inc"

void FSharpDialect::registerAttributes()
{
    addAttributes<
#define GET_ATTRDEF_LIST
#include "compiler/FSharpAttrDefs.cpp.inc"
    >();
}

//===----------------------------------------------------------------------===//
// FSharp Operations
//===----------------------------------------------------------------------===//

llvm::LogicalResult PrintOp::verify()
{
    auto firstArgType = llvm::dyn_cast<mlir::ShapedType>(getOperand(0).getType());
    if (!firstArgType || (firstArgType.getElementType() != IntegerType::get(getContext(), 8)))
    {
        mlir::emitError(getLoc(), fmt::format("expected i8 tensor type for first operand got {}",
                                              getTypeString(getOperand(0).getType()))
        );
        return llvm::failure();
    }
    return mlir::success();
}


bufferization::AliasingValueList PrintOp::getAliasingValues(::mlir::OpOperand& opOperand,
                                                            const ::mlir::bufferization::AnalysisState& state)
{
    return bufferization::AliasingValueList({
        bufferization::AliasingValue(opOperand.get(), bufferization::BufferRelation::Equivalent)
    });
}

/// Convert the given RankedTensorType into the corresponding MemRefType.
static MemRefType convertTensorToMemRef(RankedTensorType type)
{
    return MemRefType::get(type.getShape(), type.getElementType());
}

LogicalResult PrintOp::bufferize(RewriterBase& rewriter, const bufferization::BufferizationOptions& options)
{
    // 1. Rewrite tensor operands as memrefs based on type of the already
    //    bufferized callee.
    SmallVector<Value> newOperands;
    for (auto opOperand : getOperands())
    {
        // Non-tensor operands are just copied.
        if (!isa<TensorType>(opOperand.getType()))
        {
            newOperands.push_back(opOperand);
            continue;
        }

        // Retrieve buffers for tensor operands.
        FailureOr<Value> maybeBuffer =
            getBuffer(rewriter, opOperand, options);
        if (failed(maybeBuffer))
            return failure();
        Value buffer = *maybeBuffer;

        // Caller / callee type mismatch is handled with a CastOp.
        auto memRefType = convertTensorToMemRef(mlir::cast<RankedTensorType>(opOperand.getType()));
        // Since we don't yet have a clear layout story, to_memref may
        // conservatively turn tensors into more dynamic memref than necessary.
        // If the memref type of the callee fails, introduce an extra memref.cast
        // that will either canonicalize away or fail compilation until we can do
        // something better.
        if (buffer.getType() != memRefType)
        {
            assert(
                memref::CastOp::areCastCompatible(buffer.getType(), memRefType) &&
                "CallOp::bufferize: cast incompatible");
            Value castBuffer = rewriter.create<memref::CastOp>(getLoc(),
                                                               memRefType, buffer);
            buffer = castBuffer;
        }
        newOperands.push_back(buffer);
    }

    // 3. Create the new CallOp.
    Operation* newCallOp = rewriter.create<PrintOp>(
        getLoc(), newOperands);

    // 4. Replace the old op with the new op.
    bufferization::replaceOpWithBufferizedValues(rewriter, *this, newCallOp->getResults());

    return success();
}

/// A generalized parser for binary operations. This parses the different forms
/// of 'printBinaryOp' below.
static mlir::ParseResult parseBinaryOp(mlir::OpAsmParser& parser,
                                       mlir::OperationState& result)
{
    SmallVector<mlir::OpAsmParser::UnresolvedOperand, 2> operands;
    SMLoc operandsLoc = parser.getCurrentLocation();
    Type type;
    if (parser.parseOperandList(operands, /*requiredOperandCount=*/2) ||
        parser.parseOptionalAttrDict(result.attributes) ||
        parser.parseColonType(type))
        return mlir::failure();

    // If the type is a function type, it contains the input and result types of
    // this operation.
    if (FunctionType funcType = llvm::dyn_cast<FunctionType>(type))
    {
        if (parser.resolveOperands(operands, funcType.getInputs(), operandsLoc,
                                   result.operands))
            return mlir::failure();
        result.addTypes(funcType.getResults());
        return mlir::success();
    }

    // Otherwise, the parsed type is the type of both operands and results.
    if (parser.resolveOperands(operands, type, result.operands))
        return mlir::failure();
    result.addTypes(type);
    return mlir::success();
}

/// A generalized printer for binary operations. It prints in two different
/// forms depending on if all of the types match.
static void printBinaryOp(mlir::OpAsmPrinter& printer, mlir::Operation* op)
{
    printer << " " << op->getOperands();
    printer.printOptionalAttrDict(op->getAttrs());
    printer << " : ";

    // If all of the types are the same, print the type directly.
    Type resultType = *op->result_type_begin();
    if (llvm::all_of(op->getOperandTypes(),
                     [=](Type type) { return type == resultType; }))
    {
        printer << resultType;
        return;
    }

    // Otherwise, print a functional type.
    printer.printFunctionalType(op->getOperandTypes(), op->getResultTypes());
}

//===----------------------------------------------------------------------===//
// TableGen'd op method definitions
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "compiler/FSharp.cpp.inc"

std::string getTypeString(mlir::Type type)
{
    std::string typeStr;
    llvm::raw_string_ostream rso(typeStr);
    type.print(rso);
    return rso.str();
}

//===----------------------------------------------------------------------===//
// ClosureOp
//===----------------------------------------------------------------------===//

void ClosureOp::build(mlir::OpBuilder& builder, mlir::OperationState& state,
                      llvm::StringRef name, mlir::FunctionType type,
                      llvm::ArrayRef<mlir::NamedAttribute> attrs)
{
    // FunctionOpInterface provides a convenient `build` method that will populate
    // the state of our FuncOp, and create an entry block.
    buildWithEntryBlock(builder, state, name, type, attrs, type.getInputs());
}

mlir::ParseResult ClosureOp::parse(mlir::OpAsmParser& parser,
                                   mlir::OperationState& result)
{
    // Dispatch to the FunctionOpInterface provided utility method that parses the
    // function operation.
    auto buildFuncType =
        [](mlir::Builder& builder, llvm::ArrayRef<mlir::Type> argTypes,
           llvm::ArrayRef<mlir::Type> results,
           mlir::function_interface_impl::VariadicFlag,
           std::string&)
    {
        return builder.getFunctionType(argTypes, results);
    };

    return mlir::function_interface_impl::parseFunctionOp(
        parser, result, /*allowVariadic=*/false,
        getFunctionTypeAttrName(result.name), buildFuncType,
        getArgAttrsAttrName(result.name), getResAttrsAttrName(result.name));
}

void ClosureOp::print(mlir::OpAsmPrinter& p)
{
    // Dispatch to the FunctionOpInterface provided utility method that prints the
    // function operation.
    mlir::function_interface_impl::printFunctionOp(
        p, *this, /*isVariadic=*/false, getFunctionTypeAttrName(),
        getArgAttrsAttrName(), getResAttrsAttrName());
}

int ClosureOp::inferTypes()
{
    return 0;
}

void ClosureOp::assumeTypes()
{

}



//===----------------------------------------------------------------------===//
// GenericCallOp
//===----------------------------------------------------------------------===//

ClosureOp getClosureOpFromCallOp(CallOp callOp) {
    // Get the symbol reference attribute from the callee attribute
    auto calleeAttr = callOp->getAttrOfType<SymbolRefAttr>("callee");
    if (!calleeAttr)
        return nullptr;

    // Get the parent operation, which should be a module or a symbol table
    Operation *parentOp = callOp->getParentOp();
    while (parentOp && !isa<ModuleOp>(parentOp))
        parentOp = parentOp->getParentOp();

    if (!parentOp)
        return nullptr;

    // Use the SymbolTable to lookup the ClosureOp
    SymbolTable symbolTable(parentOp);
    Operation *calleeOp = symbolTable.lookup(calleeAttr.getLeafReference());
    return dyn_cast_or_null<ClosureOp>(calleeOp);
}

void CallOp::build(mlir::OpBuilder& builder, mlir::OperationState& state,
                   StringRef callee, ArrayRef<mlir::Value> arguments)
{
    // Generic call always returns an unranked Tensor initially.
    state.addTypes(UnrankedTensorType::get(builder.getF64Type()));
    state.addOperands(arguments);
    state.addAttribute("callee",
                       mlir::SymbolRefAttr::get(builder.getContext(), callee));
}

int CallOp::inferTypes()
{
    // Get the ClosureOp from the CallOp
    ClosureOp closureOp = getClosureOpFromCallOp(*this);
    if (!closureOp)
    {
        mlir::emitError(getLoc(), "Unable to find callee");
        return 0;
    }

    auto function_type = closureOp.getFunctionType();

    // If the return type of this function is specified but the return value of the called function isnt we can infer it
    // to the return type of this call
    if (!mlir::isa<NoneType>(getResult().getType()) && mlir::isa<NoneType>(function_type.getResult(0)))
    {
        // Copy the input types and set the return type to the type of the closure to this return type
        closureOp.setFunctionType(mlir::FunctionType::get(getContext(), function_type.getInputs(), getResult().getType()));
    }

    for (auto [index, input_type] : llvm::enumerate(function_type.getInputs()))
    {

    }
    return 0;
}

void CallOp::assumeTypes()
{

}

//===----------------------------------------------------------------------===//
// ArtihmeticOps
//===----------------------------------------------------------------------===//

static llvm::LogicalResult verifyArithOp(mlir::Value lhs, mlir::Value rhs)
{
    if ((mlir::isa<mlir::ShapedType>(lhs.getType()) || mlir::isa<mlir::ShapedType>(rhs.getType()))
        || (lhs.getType() != rhs.getType() && !(mlir::isa<NoneType>(lhs.getType()) || mlir::isa<
            NoneType>(rhs.getType())))
    )
    {
        mlir::emitError(lhs.getLoc(), "Expected operands to have the same scalar type or for one to be undefined.");
        return llvm::failure();
    }
    return llvm::success();
}

static mlir::Type getArithOpReturnType(const mlir::Value &lhs, const mlir::Value &rhs)
{
    if (!mlir::isa<mlir::NoneType>(lhs.getType()))
        return lhs.getType();
    return rhs.getType();
}

// Returns true if both operands have been inferred.
static int inferArithOp(Operation *op, OpOperand &lhs, OpOperand &rhs)
{
    if (mlir::isa<mlir::NoneType>(lhs.get().getType()))
        lhs.get().setType(rhs.get().getType());
    else if (mlir::isa<mlir::NoneType>(rhs.get().getType()))
        rhs.get().setType(lhs.get().getType());
    op->getResult(0).setType(lhs.get().getType());
    return mlir::isa<NoneType>(lhs.get().getType()) ? 0 : 2;
}

static void assumeArithOp(Operation *op, OpOperand &lhs, OpOperand &rhs)
{
    lhs.get().setType(IntegerType::get(op->getContext(), 32, IntegerType::SignednessSemantics::Signed));
    rhs.get().setType(lhs.get().getType());
    op->getResult(0).setType(rhs.get().getType());
}

//===----------------------------------------------------------------------===//
// AddOp
//===----------------------------------------------------------------------===//

void AddOp::build(::mlir::OpBuilder& odsBuilder, ::mlir::OperationState& odsState, Value lhs, Value rhs)
{
    odsState.addTypes(getArithOpReturnType(lhs, rhs));
    odsState.addOperands({lhs, rhs});
}

llvm::LogicalResult AddOp::verify()
{
    return verifyArithOp(getLhs(), getRhs());
}

int AddOp::inferTypes()
{
    return inferArithOp(*this, getLhsMutable(), getRhsMutable());
}

void AddOp::assumeTypes()
{
    assumeArithOp(*this, getLhsMutable(), getRhsMutable());
}

//===----------------------------------------------------------------------===//
// SubOp
//===----------------------------------------------------------------------===//

void SubOp::build(::mlir::OpBuilder& odsBuilder, ::mlir::OperationState& odsState, Value lhs, Value rhs)
{
    odsState.addTypes(getArithOpReturnType(lhs, rhs));
    odsState.addOperands({lhs, rhs});
}

llvm::LogicalResult SubOp::verify()
{
    return verifyArithOp(getLhs(), getRhs());
}

int SubOp::inferTypes()
{
    return inferArithOp(*this, getLhsMutable(), getRhsMutable());
}

void SubOp::assumeTypes()
{
    assumeArithOp(*this, getLhsMutable(), getRhsMutable());
}

//===----------------------------------------------------------------------===//
// MulOp
//===----------------------------------------------------------------------===//

void MulOp::build(::mlir::OpBuilder& odsBuilder, ::mlir::OperationState& odsState, Value lhs, Value rhs)
{
    odsState.addTypes(getArithOpReturnType(lhs, rhs));
    odsState.addOperands({lhs, rhs});
}

llvm::LogicalResult MulOp::verify()
{
    return verifyArithOp(getLhs(), getRhs());
}

int MulOp::inferTypes()
{
    return inferArithOp(*this, getLhsMutable(), getRhsMutable());
}

void MulOp::assumeTypes()
{
    assumeArithOp(*this, getLhsMutable(), getRhsMutable());
}

//===----------------------------------------------------------------------===//
// DivOp
//===----------------------------------------------------------------------===//

void DivOp::build(::mlir::OpBuilder& odsBuilder, ::mlir::OperationState& odsState, Value lhs, Value rhs)
{
    odsState.addTypes(getArithOpReturnType(lhs, rhs));
    odsState.addOperands({lhs, rhs});
}

llvm::LogicalResult DivOp::verify()
{
    return verifyArithOp(getLhs(), getRhs());
}

int DivOp::inferTypes()
{
    return inferArithOp(*this, getLhsMutable(), getRhsMutable());
}

void DivOp::assumeTypes()
{
    assumeArithOp(*this, getLhsMutable(), getRhsMutable());
}

//===----------------------------------------------------------------------===//
// ModOp
//===----------------------------------------------------------------------===//

void ModOp::build(::mlir::OpBuilder& odsBuilder, ::mlir::OperationState& odsState, Value lhs, Value rhs)
{
    odsState.addTypes(getArithOpReturnType(lhs, rhs));
    odsState.addOperands({lhs, rhs});
}

llvm::LogicalResult ModOp::verify()
{
    return verifyArithOp(getLhs(), getRhs());
}

int ModOp::inferTypes()
{
    return inferArithOp(*this, getLhsMutable(), getRhsMutable());
}

void ModOp::assumeTypes()
{
    assumeArithOp(*this, getLhsMutable(), getRhsMutable());
}

//===----------------------------------------------------------------------===//
// LogicalOps
//===----------------------------------------------------------------------===//
static llvm::LogicalResult verifyLogicalOp(mlir::Value lhs, mlir::Value rhs)
{
    if ((mlir::isa<mlir::ShapedType>(lhs.getType()) || mlir::isa<mlir::ShapedType>(rhs.getType()))
        || (lhs.getType() != rhs.getType() && !(mlir::isa<NoneType>(lhs.getType()) || mlir::isa<
            NoneType>(rhs.getType())))
    )
    {
        mlir::emitError(lhs.getLoc(), "Expected operands to have the same scalar type or for one to be undefined.");
        return llvm::failure();
    }
    return llvm::success();
}

/// Returns the type of the logical operation. If the type of the left-hand side (lhs)
/// operand is not NoneType, it returns the type of lhs. Otherwise, it returns the type
/// of the right-hand side (rhs) operand.
static mlir::Type getLogicalOpReturnType(const mlir::Value &lhs, const mlir::Value &rhs)
{
    if (!mlir::isa<mlir::NoneType>(lhs.getType()))
        return lhs.getType();
    return rhs.getType();
}

// Returns true if both operands have been inferred.
static bool inferLogicalOp(Operation *op, OpOperand &lhs, OpOperand &rhs)
{
    if (mlir::isa<mlir::NoneType>(lhs.get().getType()))
        lhs.get().setType(rhs.get().getType());
    else if (mlir::isa<mlir::NoneType>(rhs.get().getType()))
        rhs.get().setType(lhs.get().getType());
    op->getResult(0).setType(lhs.get().getType());
    return mlir::isa<NoneType>(lhs.get().getType());
}

static void assumeLogicalOp(Operation *op, OpOperand &lhs, OpOperand &rhs)
{
    lhs.get().setType(IntegerType::get(op->getContext(), 32, IntegerType::SignednessSemantics::Signed));
    rhs.get().setType(lhs.get().getType());
    op->getResult(0).setType(rhs.get().getType());
}

//===----------------------------------------------------------------------===//
// AndOp
//===----------------------------------------------------------------------===//
void AndOp::build(::mlir::OpBuilder& odsBuilder, ::mlir::OperationState& odsState, Value lhs, Value rhs)
{
    odsState.addTypes(getLogicalOpReturnType(lhs, rhs));
    odsState.addOperands({lhs, rhs});
}

llvm::LogicalResult AndOp::verify()
{
    return verifyLogicalOp(getLhs(), getRhs());
}

int AndOp::inferTypes()
{
    return inferLogicalOp(*this, getLhsMutable(), getRhsMutable());
}

void AndOp::assumeTypes()
{
    assumeLogicalOp(*this, getLhsMutable(), getRhsMutable());
}

//===----------------------------------------------------------------------===//
// OrOp
//===----------------------------------------------------------------------===//
void OrOp::build(::mlir::OpBuilder& odsBuilder, ::mlir::OperationState& odsState, Value lhs, Value rhs)
{
    odsState.addTypes(getLogicalOpReturnType(lhs, rhs));
    odsState.addOperands({lhs, rhs});
}

llvm::LogicalResult OrOp::verify()
{
    return verifyLogicalOp(getLhs(), getRhs());
}

int OrOp::inferTypes()
{
    return inferLogicalOp(*this, getLhsMutable(), getRhsMutable());
}

void OrOp::assumeTypes()
{
    assumeLogicalOp(*this, getLhsMutable(), getRhsMutable());
}

//===----------------------------------------------------------------------===//
// EqualityOps
//===----------------------------------------------------------------------===//
static llvm::LogicalResult verifyEqualityOp(mlir::Value lhs, mlir::Value rhs)
{
    if ((mlir::isa<mlir::ShapedType>(lhs.getType()) || mlir::isa<mlir::ShapedType>(rhs.getType()))
        || (lhs.getType() != rhs.getType() && !(mlir::isa<NoneType>(lhs.getType()) || mlir::isa<
            NoneType>(rhs.getType())))
    )
    {
        mlir::emitError(lhs.getLoc(), "Expected operands to have the same scalar type or for one to be undefined.");
        return llvm::failure();
    }
    return llvm::success();
}

/// Returns the type of the logical operation. If the type of the left-hand side (lhs)
/// operand is not NoneType, it returns the type of lhs. Otherwise, it returns the type
/// of the right-hand side (rhs) operand.
static mlir::Type getEqualityOpReturnType(const mlir::Value &lhs, const mlir::Value &rhs)
{
    if (!mlir::isa<mlir::NoneType>(lhs.getType()))
        return lhs.getType();
    return rhs.getType();
}

// Returns true if both operands have been inferred.
static bool inferEqualityOp(Operation *op, OpOperand &lhs, OpOperand &rhs)
{
    if (mlir::isa<mlir::NoneType>(lhs.get().getType()))
        lhs.get().setType(rhs.get().getType());
    else if (mlir::isa<mlir::NoneType>(rhs.get().getType()))
        rhs.get().setType(lhs.get().getType());
    op->getResult(0).setType(lhs.get().getType());
    return mlir::isa<NoneType>(lhs.get().getType());
}

static void assumeEqualityOp(Operation *op, OpOperand &lhs, OpOperand &rhs)
{
    lhs.get().setType(IntegerType::get(op->getContext(), 32, IntegerType::SignednessSemantics::Signed));
    rhs.get().setType(lhs.get().getType());
    op->getResult(0).setType(rhs.get().getType());
}

//===----------------------------------------------------------------------===//
// EqualOp
//===----------------------------------------------------------------------===//
void EqualOp::build(::mlir::OpBuilder& odsBuilder, ::mlir::OperationState& odsState, Value lhs, Value rhs)
{
    odsState.addTypes(getEqualityOpReturnType(lhs, rhs));
    odsState.addOperands({lhs, rhs});
}

llvm::LogicalResult EqualOp::verify()
{
    return verifyEqualityOp(getLhs(), getRhs());
}

int EqualOp::inferTypes()
{
    return inferEqualityOp(*this, getLhsMutable(), getRhsMutable());
}

void EqualOp::assumeTypes()
{
    assumeEqualityOp(*this, getLhsMutable(), getRhsMutable());
}

//===----------------------------------------------------------------------===//
// NotEqualOp
//===----------------------------------------------------------------------===//
void NotEqualOp::build(::mlir::OpBuilder& odsBuilder, ::mlir::OperationState& odsState, Value lhs, Value rhs)
{
    odsState.addTypes(getEqualityOpReturnType(lhs, rhs));
    odsState.addOperands({lhs, rhs});
}

llvm::LogicalResult NotEqualOp::verify()
{
    return verifyEqualityOp(getLhs(), getRhs());
}

int NotEqualOp::inferTypes()
{
    return inferEqualityOp(*this, getLhsMutable(), getRhsMutable());
}

void NotEqualOp::assumeTypes()
{
    assumeEqualityOp(*this, getLhsMutable(), getRhsMutable());
}
