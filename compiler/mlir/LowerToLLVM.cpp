//
// Created by lasse on 1/14/25.
//

#include "compiler/FSharpPasses.h"

#include <mlir/Dialect/MemRef/IR/MemRef.h>

#include "compiler/FSharpDialect.h"

#include <mlir/IR/PatternMatch.h>

#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/FunctionCallUtils.h"

using namespace mlir;

/// Generic print function lookupOrCreate helper.
LLVM::LLVMFuncOp lookupOrCreateFn(PatternRewriter& rewriter,
                                  ModuleOp moduleOp,
                                  StringRef name,
                                  ArrayRef<Type> paramTypes,
                                  Type resultType, bool isVarArg)
{
    assert(moduleOp->hasTrait<OpTrait::SymbolTable>() &&
        "expected SymbolTable operation");
    auto func = llvm::dyn_cast_or_null<LLVM::LLVMFuncOp>(
        SymbolTable::lookupSymbolIn(moduleOp, name));
    if (func)
        return func;
    return rewriter.create<LLVM::LLVMFuncOp>(
        moduleOp->getLoc(), name,
        LLVM::LLVMFunctionType::get(resultType, paramTypes, isVarArg));
}


namespace
{
    /// Lowers "fsharp.print" to a printf call.
    class PrintOpLowering : public ConversionPattern
    {
    public:
        explicit PrintOpLowering(MLIRContext* context)
            : ConversionPattern(fsharp::PrintOp::getOperationName(), 1, context)
        {
        }

        LogicalResult matchAndRewrite(Operation* op, ArrayRef<Value> operands, ConversionPatternRewriter& rewriter) const override
        {
            auto *context = rewriter.getContext();
            auto memRefType = llvm::cast<MemRefType>((*op->operand_type_begin()));
            auto memRefShape = memRefType.getShape();
            auto loc = op->getLoc();

            auto parentModule = op->getParentOfType<ModuleOp>();

            auto printfRef = getOrInsertPrinf(rewriter, parentModule);
            // Generate a call to printf for the current element of the loop.
            auto printOp = cast<fsharp::PrintOp>(op);
        }

    private:
        static mlir::FlatSymbolRefAttr getOrInsertPrinf(PatternRewriter& rewriter,
                                                        ModuleOp module)
        {
            auto* context = module.getContext();

            // Insert the printf function into the body of the parent module.
            PatternRewriter::InsertionGuard insertGuard(rewriter);
            rewriter.setInsertionPointToStart(module.getBody());
            lookupOrCreateFn(rewriter,
                             module,
                             "prinf",
                             LLVM::LLVMPointerType::get(context, 8),
                             IntegerType::get(context, 32), true);
            return SymbolRefAttr::get(context, "printf");
        }
    };
}
