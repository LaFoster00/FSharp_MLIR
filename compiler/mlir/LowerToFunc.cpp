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

//===----------------------------------------------------------------------===//
// LowerToFunc RewritePatterns: Closure operations
//===----------------------------------------------------------------------===//

struct ClosureOpLowering : public OpConversionPattern<fsharp::ClosureOp>
{
    using OpConversionPattern<fsharp::ClosureOp>::OpConversionPattern;

    LogicalResult matchAndRewrite(fsharp::ClosureOp closure_op, OpAdaptor adaptor,
                                  ConversionPatternRewriter& rewriter) const final
    {
        OpBuilder builder(closure_op);
        auto loc = closure_op.getLoc();

        // Extract the explicit function type.
        FunctionType funcType = closure_op.getFunctionType();

        // Traverse the closure body to identify implicit captures.
        SetVector<Value> implicitCaptures;
        closure_op.getBody().walk([&](Operation* op)
        {
            for (Value operand : op->getOperands())
            {
                // Skip values already defined inside the closure region.
                if (closure_op->isAncestor(operand.getParentRegion()->getParentOp()))
                    continue;

                // Add operands from the outer scope as implicit captures.
                implicitCaptures.insert(operand);
            }
        });

        // Build the argument types for the generated function.
        SmallVector<Type, 4> capturedArgTypes, explicitArgTypes;
        for (Value capture : implicitCaptures)
            capturedArgTypes.push_back(capture.getType());
        explicitArgTypes.append(funcType.getInputs().begin(), funcType.getInputs().end());

        SmallVector<Type, 4> allArgTypes(capturedArgTypes);
        allArgTypes.append(explicitArgTypes.begin(), explicitArgTypes.end());

        // Create the new function type with captures and explicit arguments.
        auto newFuncType = builder.getFunctionType(allArgTypes, funcType.getResults());


        // Generate a unique symbol name for the new function.
        std::string funcName;
        if (closure_op.getSymName() != "main")
            funcName = (closure_op.getSymName() + "_lowered").str();
        else
            funcName = "main";

        // Create the new `func.func` operation.
        auto funcOp = builder.create<func::FuncOp>(loc, funcName, newFuncType);

        // Inline the closure body into the function region.
        Region& funcRegion = funcOp.getBody();
        rewriter.inlineRegionBefore(closure_op.getBody(), funcRegion, funcRegion.end());
        
        // Replace all references to the closure arguments with the new function arguments.
        for (auto [index, old_arg] : llvm::enumerate(closure_op.getArguments()))
        {
            old_arg.replaceAllUsesWith(funcOp.getArgument(index));
        }

        // Replace `ClosureOp` with a symbol reference to the new function.
        builder.setInsertionPointAfter(closure_op);
        closure_op->getParentOp()->walk([&](fsharp::CallOp call_op)
        {
            if (call_op.getCallee() == closure_op.getSymName())
            {
                OpBuilder builder(call_op);
                auto loc = call_op.getLoc();

                // Get the arguments for the CallOp
                SmallVector<Value, 4> callArgs = call_op.getOperands();

                // Create a new func::CallOp with the new function name
                auto newCallOp = builder.create<fsharp::CallOp>(loc, funcName, call_op->getResultTypes(), callArgs);

                // Replace the old CallOp with the new CallOp
                call_op.replaceAllUsesWith(newCallOp.getResult());
                call_op.erase();
            }
        });
        rewriter.eraseOp(closure_op);
        return success();
    }
};

//===----------------------------------------------------------------------===//
// LowerToFunc RewritePatterns: Call operations
//===----------------------------------------------------------------------===//

/// Find a closure with the given name in the current scope or parent scopes.
mlir::fsharp::ClosureOp findClosureInScope(mlir::Operation *startOp, mlir::StringRef closureName) {
    mlir::Operation *currentOp = startOp;

    // Traverse up through parent operations (or regions) to find the closure
    while (currentOp) {
        // Check if the current operation has a SymbolTable
        if (currentOp->hasTrait<mlir::OpTrait::SymbolTable>()) {
            // Try to lookup the closure in the current SymbolTable
            mlir::Operation *closure = mlir::SymbolTable::lookupSymbolIn(currentOp, closureName);
            if (auto closure_op = mlir::dyn_cast<mlir::fsharp::ClosureOp>(closure)) {
                return closure_op; // Found the closure
            }
        }

        // Move to the parent operation
        currentOp = currentOp->getParentOp();
    }

    // If no closure was found, return nullptr
    return nullptr;
}

struct CallOpLowering : public OpConversionPattern<fsharp::CallOp>
{
    using OpConversionPattern<fsharp::CallOp>::OpConversionPattern;

    LogicalResult matchAndRewrite(fsharp::CallOp call_op, OpAdaptor adaptor,
                                  ConversionPatternRewriter& rewriter) const final
    {
        OpBuilder builder(call_op);
        auto loc = call_op.getLoc();

        // Get the symbol of the callee.
        auto calleeSymbol = call_op.getCalleeAttr();
        auto funcOp = findClosureInScope(call_op, calleeSymbol.getValue());
        if (!funcOp)
        {
            call_op.emitError("Callee function not found in symbol table");
            return failure();
        }

        // Get the arguments for the `CallOp`.
        SmallVector<Value, 4> callArgs = call_op.getOperands();

        // Replace `CallOp` with `func.call`.
        auto call = builder.create<func::CallOp>(loc, funcOp.getName(), funcOp.getFunctionType().getResults(),
                                                 callArgs);
        call_op->replaceAllUsesWith(call.getResults());
        rewriter.eraseOp(call_op);
        return success();
    }
};

//===----------------------------------------------------------------------===//
// FSharpToFuncLoweringPass
//===----------------------------------------------------------------------===//

/// This is a partial lowering to func ops for the fsharp operations
namespace
{
    struct FSharpToFuncLoweringPass
        : public PassWrapper<FSharpToFuncLoweringPass, OperationPass<ModuleOp>>
    {
        MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(FSharpToFuncLoweringPass)

        void getDependentDialects(DialectRegistry& registry) const override
        {
            registry.insert<affine::AffineDialect, func::FuncDialect,
                            memref::MemRefDialect>();
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
    target.addIllegalOp<fsharp::ClosureOp>();
    target.addIllegalOp<fsharp::CallOp>();

    RewritePatternSet patterns(&getContext());

    // We only want to lower the 'fsharp.func' operations
    patterns.add<ClosureOpLowering>(&getContext());
    patterns.add<CallOpLowering>(&getContext());

    // We want to completely lower to LLVM, so we use a `FullConversion`. This
    // ensures that only legal operations will remain after the conversion.
    auto module = getOperation();
    if (failed(applyFullConversion(module, target, std::move(patterns))))
        signalPassFailure();
}

namespace mlir::fsharp
{
    std::unique_ptr<mlir::Pass> createLowerToFunctionPass()
    {
        return std::make_unique<FSharpToFuncLoweringPass>();
    }
}
