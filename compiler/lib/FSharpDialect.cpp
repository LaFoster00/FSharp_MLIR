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
#include <string>
#include <fmt/format.h>

using namespace mlir;
using namespace mlir::fsharp;


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

mlir::Type getMLIRType(mlir::OpBuilder b, const std::string& type_name)
{
    if (type_name == "int")
        return b.getI32Type();
    if (type_name == "float")
        return b.getF32Type();
    if (type_name == "bool")
        return b.getI8Type();
    if (type_name == "string")
        return mlir::UnrankedTensorType::get(b.getI8Type());
    assert(false && "Type not supported!");
}
