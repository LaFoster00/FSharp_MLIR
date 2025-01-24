//
// Created by lasse on 10/01/2025.
//

#include "compiler/FSharpDialect.h"

#include <compiler/CompilerUtils.h>

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


//===----------------------------------------------------------------------===//
// PrintOp
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
static MemRefType convertTensorToMemRef(TensorType type)
{
    if (auto unranked = mlir::dyn_cast<UnrankedTensorType>(type))
        return MemRefType::get({1}, unranked.getElementType());
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
        auto memRefType = convertTensorToMemRef(mlir::cast<TensorType>(opOperand.getType()));
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

void ClosureOp::inferFromOperands()
{
    auto& entry_block = front();
    for (auto [i, arg] : llvm::enumerate(entry_block.getArguments()))
    {
        arg.setType(getArgumentTypes()[i]);
    }

    // In case the return operation has a typed operand but the function doesn't, copy it from the return operation.
    updateSignatureFromBody();
}

void ClosureOp::inferFromReturnType()
{
}

void ClosureOp::inferFromUnknown()
{
}

//===----------------------------------------------------------------------===//
// ReturnOp
//===----------------------------------------------------------------------===//

LogicalResult fsharp::ReturnOp::verify()
{
    auto function = cast<ClosureOp>((*this)->getParentOp());

    // The operand number and types must match the function signature.
    const auto& results = function.getFunctionType().getResults();
    if (getNumOperands() != results.size())
        return emitOpError("has ")
            << getNumOperands() << " operands, but enclosing function (@"
            << function.getName() << ") returns " << results.size();

    for (unsigned i = 0, e = results.size(); i != e; ++i)
        if (getOperand().getType() != results[i])
            return emitError() << "type of return operand " << i << " ("
                << getOperand().getType()
                << ") doesn't match function result type ("
                << results[i] << ")"
                << " in function @" << function.getName();

    return success();
}


//===----------------------------------------------------------------------===//
// CallOp
//===----------------------------------------------------------------------===//

/// Find a closure with the given name in the current scope or parent scopes.
mlir::fsharp::ClosureOp findClosureInScope(mlir::Operation* startOp, mlir::StringRef closureName)
{
    mlir::Operation* currentOp = startOp;

    // Traverse up through parent operations (or regions) to find the closure
    while (currentOp)
    {
        // Check if the current operation has a SymbolTable
        if (currentOp->hasTrait<mlir::OpTrait::SymbolTable>())
        {
            // Try to lookup the closure in the current SymbolTable
            mlir::Operation* closure = mlir::SymbolTable::lookupSymbolIn(currentOp, closureName);
            if (auto closure_op = mlir::dyn_cast<mlir::fsharp::ClosureOp>(closure))
            {
                return closure_op; // Found the closure
            }
        }

        // Move to the parent operation
        currentOp = currentOp->getParentOp();
    }

    // If no closure was found, return nullptr
    return nullptr;
}

void ClosureOp::updateSignatureFromBody()
{
    mlir::SmallVector<mlir::Type, 4> inputs(getFunctionType().getInputs());
    for (auto [i, block_arg] : llvm::enumerate(front().getArguments()))
    {
        inputs[i] = block_arg.getType();
    }
    mlir::SmallVector<Type, 4> results(getFunctionType().getResults());
    walk<WalkOrder::PreOrder>([&](fsharp::ReturnOp return_op)
    {
        for (auto [i, operand_type] : llvm::enumerate(return_op.getOperandTypes()))
        {
            if (results.size() > i)
                results[i] = operand_type;
            else
                mlir::emitError(getLoc(), "Function return type mismatch!");
        }
        return mlir::WalkResult::interrupt();
    });

    setType(mlir::FunctionType::get(getContext(), inputs, results));
}


//===----------------------------------------------------------------------===//
// ReturnOp
//===----------------------------------------------------------------------===//

// This is the only one that actually matters for our purposes since we only care about the type of the returned object
void ReturnOp::inferFromOperands()
{
    auto previous_function_type = getParentOp().getFunctionType();
    if (!previous_function_type)
    {
        mlir::emitError(getLoc(), "Cant access parent closure op!");
        return;
    }

    getParentOp().setFunctionType(
        mlir::FunctionType::get(getContext(), previous_function_type.getInputs(), getOperandTypes()));
    // Now that the function type has been updated we need to update the types of the call ops that use this function
    auto function_uses = getParentOp().getSymbolUses(getParentOp()->getParentOp());
    if (function_uses.has_value())
    {
        for (auto symbol_use : function_uses.value())
        {
            auto user = symbol_use.getUser();
            if (auto call_op = mlir::dyn_cast<CallOp>(user))
            {
                call_op.getResult(0).setType(getOperand().getType());
            }
        }
    }
}

// Return ops wont be resolved by this step since they take their type from the returned object which is not known at this point
void ReturnOp::inferFromReturnType()
{
}

void ReturnOp::inferFromUnknown()
{
    if (isa<NoneType>(getOperand().getType()))
    {
        getOperand().setType(IntegerType::get(getContext(), 32, IntegerType::Signed));
    }
    getParentOp().updateSignatureFromBody();
}


void CallOp::inferFromOperands()
{
    // Get the ClosureOp from the CallOp

    ClosureOp closureOp = findClosureInScope(*this, getCallee());
    if (!closureOp)
    {
        mlir::emitError(getLoc(), "Unable to find callee");
        return;
    }

    auto closure_type = closureOp.getFunctionType();

    auto operand_types = getOperandTypes();
    // In case we infer any of the types of the targeted closure we need to update the closure types inputs later on
    llvm::SmallVector<mlir::Type, 4> new_closure_inputs(closure_type.getInputs());
    for (auto [index, operand_type] : llvm::enumerate(operand_types))
    {
        // If the operand types are not the same as the input types of the closure we can substitute them for this calls types
        if (new_closure_inputs[index] != operand_type)
        {
            new_closure_inputs[index] = operand_type;
        }
    }
    closureOp.setFunctionType(mlir::FunctionType::get(getContext(), new_closure_inputs, closure_type.getResults()));
}

void CallOp::inferFromReturnType()
{
}


// In case one of the arguments for this call is not inferred yet, we check if the callee is infered and change the
// type of the arguments for the call to match the callee.
void CallOp::inferFromUnknown()
{
    ClosureOp closureOp = findClosureInScope(*this, getCallee());
    if (!closureOp)
    {
        mlir::emitError(getLoc(), "Unable to find callee");
        return;
    }
    auto closure_type = closureOp.getFunctionType();

    auto operands = getOperands();
    for (auto [index, closure_input_type] : llvm::enumerate(closure_type.getInputs()))
    {
        if (isa<NoneType>(operands[index].getType()))
        {
            operands[index].setType(closure_input_type);
        }
    }
    closureOp.updateSignatureFromBody();
}

//===----------------------------------------------------------------------===//
// Binary Operations
//===----------------------------------------------------------------------===//

static llvm::LogicalResult verifyBinaryOp(mlir::Value lhs, mlir::Value rhs)
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

static mlir::Type getBinaryOpReturnType(const mlir::Value& lhs, const mlir::Value& rhs)
{
    if (!mlir::isa<mlir::NoneType>(lhs.getType()))
        return lhs.getType();
    return rhs.getType();
}

// Returns true if both operands have been inferred.
static void inferBinaryOpFromOperands(Operation* op, OpOperand& lhs, OpOperand& rhs)
{
    op->getResult(0).setType(lhs.get().getType());
}

static void inferBinaryOpFromResultType(Operation* op, OpOperand& lhs, OpOperand& rhs)
{
    if (mlir::isa<NoneType>(lhs.get().getType()))
    {
        lhs.get().setType(op->getResultTypes()[0]);
    }
    if (mlir::isa<NoneType>(rhs.get().getType()))
    {
        rhs.get().setType(op->getResultTypes()[0]);
    }
    // Update the surrounding closure so that its input args match with the block args of the closure region. //TODO this is a bit hacky
    if (auto closure = mlir::dyn_cast<ClosureOp>(op->getParentOp()))
    {
        closure.updateSignatureFromBody();
    }
}

static void assumeBinaryOp(Operation* op, OpOperand& lhs, OpOperand& rhs)
{
    lhs.get().setType(IntegerType::get(op->getContext(), 32, IntegerType::Signed));
    rhs.get().setType(lhs.get().getType());
    op->getResult(0).setType(rhs.get().getType());
    // Update the surrounding closure so that its input args match with the block args of the closure region. //TODO this is a bit hacky
    if (auto closure = mlir::dyn_cast<ClosureOp>(op->getParentOp()))
    {
        closure.updateSignatureFromBody();
    }
}

//===----------------------------------------------------------------------===//
// ArtihmeticOps
//===----------------------------------------------------------------------===//

#define GENERATE_BINARY_OP_BUILDER(op_name) \
    void op_name::build(::mlir::OpBuilder& odsBuilder, ::mlir::OperationState& odsState, Value lhs, Value rhs) \
    { \
        odsState.addTypes(getBinaryOpReturnType(lhs, rhs)); \
        odsState.addOperands({lhs, rhs}); \
    }

#define GENERATE_BINARY_OP_VERIFIER(op_name) \
    llvm::LogicalResult op_name::verify() \
    { \
        return verifyBinaryOp(getLhs(), getRhs()); \
    }

#define GENERATE_BINARY_OP_INFERENCE(op_name) \
    void op_name::inferFromOperands() \
    { \
        inferBinaryOpFromOperands(*this, getLhsMutable(), getRhsMutable()); \
    } \
    void op_name::inferFromReturnType() \
    { \
        inferBinaryOpFromResultType(*this, getLhsMutable(), getRhsMutable()); \
    } \
    void op_name::inferFromUnknown() \
    { \
        assumeBinaryOp(*this, getLhsMutable(), getRhsMutable()); \
    }

#define GENERATE_BINARY_OP(op_name) \
    GENERATE_BINARY_OP_BUILDER(op_name) \
    GENERATE_BINARY_OP_VERIFIER(op_name) \
    GENERATE_BINARY_OP_INFERENCE(op_name)

//===----------------------------------------------------------------------===//
// AddOp
//===----------------------------------------------------------------------===//

GENERATE_BINARY_OP(AddOp)

//===----------------------------------------------------------------------===//
// SubOp
//===----------------------------------------------------------------------===//

GENERATE_BINARY_OP(SubOp)

//===----------------------------------------------------------------------===//
// MulOp
//===----------------------------------------------------------------------===//

GENERATE_BINARY_OP(MulOp)

//===----------------------------------------------------------------------===//
// DivOp
//===----------------------------------------------------------------------===//

GENERATE_BINARY_OP(DivOp)

//===----------------------------------------------------------------------===//
// ModOp
//===----------------------------------------------------------------------===//

GENERATE_BINARY_OP(ModOp)


//===----------------------------------------------------------------------===//
// EqualityOps
//===----------------------------------------------------------------------===//

// Equality ops always return boolean
static void inferEqualityOpFromOperands(Operation* op, OpOperand& lhs, OpOperand& rhs)
{
}

// Skip inference for equality ops since they should not influence the types that get resolved. After all the output
// of a equality operation doesn't tell us anything about the operands
static void inferEqualityOpFromResultType(Operation* op, OpOperand& lhs, OpOperand& rhs)
{
}

static void assumeEqualityOp(Operation* op, OpOperand& lhs, OpOperand& rhs)
{
    lhs.get().setType(IntegerType::get(op->getContext(), 32, IntegerType::Signed));
    rhs.get().setType(lhs.get().getType());
    // Update the surrounding closure so that its input args match with the block args of the closure region.
    if (auto closure = mlir::dyn_cast<ClosureOp>(op->getParentOp()))
    {
        closure.updateSignatureFromBody();
    }
}

#define GENERATE_EQUALITY_OP_BUILDER(op_name) \
    void op_name::build(::mlir::OpBuilder& odsBuilder, ::mlir::OperationState& odsState, Value lhs, Value rhs) \
    { \
        odsState.addTypes(IntegerType::get(odsBuilder.getContext(), 1)); \
        odsState.addOperands({lhs, rhs}); \
    }

#define GENERATE_EQUALITY_OP_VERIFIER(op_name) \
    llvm::LogicalResult op_name::verify() \
    { \
        return verifyBinaryOp(getLhs(), getRhs()); \
    }

#define GENERATE_EQUALITY_OP_INFERENCE(op_name) \
    void op_name::inferFromOperands() \
    { \
        inferEqualityOpFromOperands(*this, getLhsMutable(), getRhsMutable()); \
    } \
    void op_name::inferFromReturnType() \
    { \
        inferEqualityOpFromResultType(*this, getLhsMutable(), getRhsMutable()); \
    } \
    void op_name::inferFromUnknown() \
    { \
        assumeEqualityOp(*this, getLhsMutable(), getRhsMutable()); \
    }

#define GENERATE_EQUALITY_OP(op_name) \
    GENERATE_EQUALITY_OP_BUILDER(op_name) \
    GENERATE_EQUALITY_OP_VERIFIER(op_name) \
    GENERATE_EQUALITY_OP_INFERENCE(op_name)

//===----------------------------------------------------------------------===//
// EqualOp
//===----------------------------------------------------------------------===//

GENERATE_EQUALITY_OP(EqualOp)

//===----------------------------------------------------------------------===//
// NotEqualOp
//===----------------------------------------------------------------------===//

GENERATE_EQUALITY_OP(NotEqualOp)


//===----------------------------------------------------------------------===//
// RelationOps
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// LessOp
//===----------------------------------------------------------------------===//

GENERATE_EQUALITY_OP(LessOp)

//===----------------------------------------------------------------------===//
// LessEqualOp
//===----------------------------------------------------------------------===//

GENERATE_EQUALITY_OP(LessEqualOp)

//===----------------------------------------------------------------------===//
// GreaterOp
//===----------------------------------------------------------------------===//

GENERATE_EQUALITY_OP(GreaterOp)

//===----------------------------------------------------------------------===//
// GreaterEqualOp
//===----------------------------------------------------------------------===//

GENERATE_EQUALITY_OP(GreaterEqualOp)


//===----------------------------------------------------------------------===//
// LogicalOps
//===----------------------------------------------------------------------===//

// Logical Ops always return boolean
static void inferLogicalOpFromOperands(Operation* op, OpOperand& lhs, OpOperand& rhs)
{
}

// Same as equality
static void inferLogicalOpFromResultType(Operation* op, OpOperand& lhs, OpOperand& rhs)
{
    if (mlir::isa<NoneType>(lhs.get().getType()))
    {
        lhs.get().setType(mlir::IntegerType::get(op->getContext(), 1));
    }
    if (mlir::isa<NoneType>(rhs.get().getType()))
    {
        rhs.get().setType(mlir::IntegerType::get(op->getContext(), 1));
    }
    // Update the surrounding closure so that its input args match with the block args of the closure region. //TODO this is a bit hacky
    if (auto closure = mlir::dyn_cast<ClosureOp>(op->getParentOp()))
    {
        closure.updateSignatureFromBody();
    }
}

static void assumeLogicalOp(Operation* op, OpOperand& lhs, OpOperand& rhs)
{
    lhs.get().setType(IntegerType::get(op->getContext(), 1));
    rhs.get().setType(lhs.get().getType());
    // Update the surrounding closure so that its input args match with the block args of the closure region.
    if (auto closure = mlir::dyn_cast<ClosureOp>(op->getParentOp()))
    {
        closure.updateSignatureFromBody();
    }
}

static llvm::LogicalResult verifyLogicalOp(mlir::Value lhs, mlir::Value rhs)
{
    bool lhs_correct = false;
    bool rhs_correct = false;
    if (auto lhsType = mlir::dyn_cast<IntegerType>(lhs.getType()))
    {
        if (lhsType.getWidth() != 1)
        {
            mlir::emitError(lhs.getLoc(), "Expected first operand to have type bool.");
            return llvm::failure();
        }
        lhs_correct = true;
    }
    if (auto rhsType = mlir::dyn_cast<IntegerType>(lhs.getType()))
    {
        if (rhsType.getWidth() != 1)
        {
            mlir::emitError(lhs.getLoc(), "Expected first operand to have type bool.");
            return llvm::failure();
        }
        rhs_correct = true;
    }

    if (!lhs_correct || mlir::isa<NoneType>(lhs.getType()))
        lhs_correct = true;

    if (!rhs_correct || mlir::isa<NoneType>(rhs.getType()))
        rhs_correct = true;

    if (lhs_correct || rhs_correct)
        return llvm::success();

    mlir::emitError(lhs.getLoc(), "Expected operands to have type bool or to be unspecified.");
    return llvm::failure();
}

#define GENERATE_LOGICAL_OP_BUILDER(op_name) \
    void op_name::build(::mlir::OpBuilder& odsBuilder, ::mlir::OperationState& odsState, Value lhs, Value rhs) \
    { \
        odsState.addTypes(IntegerType::get(odsBuilder.getContext(), 1)); \
        odsState.addOperands({lhs, rhs}); \
    }

#define GENERATE_LOGICAL_OP_VERIFIER(op_name) \
    llvm::LogicalResult op_name::verify() \
    { \
        return verifyLogicalOp(getLhs(), getRhs()); \
    }

#define GENERATE_LOGICAL_OP_INFERENCE(op_name) \
    void op_name::inferFromOperands() \
    { \
        inferLogicalOpFromOperands(*this, getLhsMutable(), getRhsMutable()); \
    } \
    void op_name::inferFromReturnType() \
    { \
        inferLogicalOpFromResultType(*this, getLhsMutable(), getRhsMutable()); \
    } \
    void op_name::inferFromUnknown() \
    { \
        assumeLogicalOp(*this, getLhsMutable(), getRhsMutable()); \
    }

#define GENERATE_LOGICAL_OP(op_name) \
    GENERATE_LOGICAL_OP_BUILDER(op_name) \
    GENERATE_LOGICAL_OP_VERIFIER(op_name) \
    GENERATE_LOGICAL_OP_INFERENCE(op_name)

//===----------------------------------------------------------------------===//
// AndOp
//===----------------------------------------------------------------------===//

GENERATE_LOGICAL_OP(AndOp)

//===----------------------------------------------------------------------===//
// OrOp
//===----------------------------------------------------------------------===//

GENERATE_LOGICAL_OP(OrOp)
