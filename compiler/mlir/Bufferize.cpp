//
// Created by lasse on 1/14/25.
//

#include "compiler/FSharpPasses.h"
#include "compiler/FSharpDialect.h"

#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;

/// Convert the given RankedTensorType into the corresponding MemRefType.
static MemRefType convertTensorToMemRef(RankedTensorType type)
{
    return MemRefType::get(type.getShape(), type.getElementType());
}

/// Insert an allocation and deallocation for the given MemRefType.
static Value insertAllocAndDealloc(MemRefType type, Location loc,
                                   PatternRewriter& rewriter)
{
    auto alloc = rewriter.create<memref::AllocOp>(loc, type);

    // Make sure to allocate at the beginning of the block.
    auto* parentBlock = alloc->getBlock();
    alloc->moveBefore(&parentBlock->front());

    // Make sure to deallocate this alloc at the end of the block. This is fine
    // as fsharp functions have no control flow.
    auto dealloc = rewriter.create<memref::DeallocOp>(loc, alloc);
    dealloc->moveBefore(&parentBlock->back());
    return alloc;
}
