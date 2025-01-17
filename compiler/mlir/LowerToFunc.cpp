//
// Created by lasse on 17/01/2025.
//

#include "compiler/FSharpDialect.h"
#include "compiler/FSharpPasses.h"

#include "mlir/Dialect/LLVMIR/LLVMAttrs.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/TypeID.h"

#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVM.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVMPass.h"
#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include <mlir/Dialect/LLVMIR/FunctionCallUtils.h>
#include <mlir/Dialect/Affine/IR/AffineOps.h>
#include <mlir/IR/BuiltinDialect.h>
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/Support/Casting.h"
#include <memory>
#include <utility>


using namespace mlir;

namespace
{
    /// Lowers fsharp.func to func.func operations.
    class FuncOpLowering : public mlir::ConversionPattern
    {
    public:
        explicit FuncOpLowering(MLIRContext* context)
            : ConversionPattern(fsharp::FuncOp::getOperationName(), 1, context)
        {
        }

        LogicalResult matchAndRewrite(Operation* op, ArrayRef<Value> operands,
                                      ConversionPatternRewriter& rewriter) const override
        {
        }
    };
}

//===----------------------------------------------------------------------===//
// FSharpToFuncLoweringPass
//===----------------------------------------------------------------------===//

namespace
{
    struct FSharpToFuncLoweringPass
        : public PassWrapper<FSharpToFuncLoweringPass, OperationPass<ModuleOp>>
    {
        MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(FSharpToFuncLoweringPass)

        void getDependentDialects(DialectRegistry& registry) const override
        {
            registry.insert<LLVM::LLVMDialect, scf::SCFDialect>();
        }

        void runOnOperation() final;
    };
} // namespace

void FSharpToFuncLoweringPass::runOnOperation()
{
    // The first thing to define is the conversion target. This will define the
    // final target for this lowering. For this lowering, we are targeting all dialects excluding the fsharp.func op.
    mlir::ConversionTarget target(getContext());
    // We define the specific operations, or dialects, that are legal targets for
    // this lowering. In our case, we are lowering to a combination of the
    // `Affine`, `Arith`, `Func`, and `MemRef` dialects.
    target.addLegalDialect<affine::AffineDialect, mlir::BuiltinDialect,
                           arith::ArithDialect, func::FuncDialect,
                           memref::MemRefDialect, fsharp::FSharpDialect>();
    target.addIllegalOp<fsharp::FuncOp>();

    RewritePatternSet patterns(&getContext());

    // We only want to lower the 'fsharp.func' operations
    patterns.add<FuncOpLowering>(&getContext());

    // We want to completely lower to LLVM, so we use a `FullConversion`. This
    // ensures that only legal operations will remain after the conversion.
    auto module = getOperation();
    if (failed(applyFullConversion(module, target, std::move(patterns))))
        signalPassFailure();
}

std::unique_ptr<mlir::Pass> createLowerToFunctionPass()
{
    return std::make_unique<FSharpToFuncLoweringPass>();
}
