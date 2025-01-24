//
// Created by lasse on 17/01/2025.
//

#include "compiler/FSharpDialect.h"
#include "compiler/FSharpPasses.h"

#include "compiler/CompilerUtils.h"

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
        FunctionType old_func_type = closure_op.getFunctionType();

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
        if (!implicitCaptures.empty())
        {
            mlir::emitError(closure_op.getLoc(), "\nImplicit captures not supported yet: \n") << *closure_op;
            return failure();
        }

        // Build the argument types for the generated function. // TODO implicit captures should be captured as a struct
        SmallVector<Type, 4> capturedArgTypes, explicitArgTypes;
        for (Value capture : implicitCaptures)
            capturedArgTypes.push_back(capture.getType());
        explicitArgTypes.append(old_func_type.getInputs().begin(), old_func_type.getInputs().end());

        SmallVector<Type, 4> allArgTypes(capturedArgTypes);
        allArgTypes.append(explicitArgTypes.begin(), explicitArgTypes.end());

        // Create the new function type with captures and explicit arguments.
        builder.setInsertionPointToStart(closure_op->getParentOfType<ModuleOp>().getBody());
        auto new_func_type = builder.getFunctionType(allArgTypes, old_func_type.getResults());

        // Generate a unique symbol name for the new function.
        static int CLOSURE_COUNTER = 0;
        std::string lower_func_name;
        if (closure_op.getSymName() != "main")
            lower_func_name = (closure_op.getSymName() + "_lowered_" + std::to_string(CLOSURE_COUNTER++)).str();
        else
            lower_func_name = "main";

        // Create the new `func.func` operation.
        auto new_func_op = builder.create<func::FuncOp>(loc, lower_func_name, new_func_type);

        // Inline the closure body into the function region.
        Region& new_func_region = new_func_op.getBody();
        rewriter.inlineRegionBefore(closure_op.getBody(), new_func_region, new_func_region.end());

        // Replace all references to the closure arguments with the new function arguments.
        for (auto [index, old_arg] : llvm::enumerate(closure_op.getArguments()))
        {
            old_arg.replaceAllUsesWith(new_func_op.getArgument(index));
        }

        if (failed(SymbolTable::replaceAllSymbolUses(closure_op.getSymNameAttr(),
                                                     new_func_op.getSymNameAttr(),
                                                     closure_op->getParentOp())))
        {
            return failure();
        }
        /*
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
        });*/
        rewriter.eraseOp(closure_op);
        return success();
    }
};

//===----------------------------------------------------------------------===//
// LowerToFunc RewritePatterns: Return operations
//===----------------------------------------------------------------------===//

struct ReturnOpLowering : public OpConversionPattern<fsharp::ReturnOp>
{
    using OpConversionPattern<fsharp::ReturnOp>::OpConversionPattern;

    LogicalResult matchAndRewrite(fsharp::ReturnOp return_op, OpAdaptor adaptor,
                                  ConversionPatternRewriter& rewriter) const final
    {
        rewriter.setInsertionPointAfter(return_op);
        rewriter.create<func::ReturnOp>(return_op.getLoc(), return_op.getOperands());
        rewriter.eraseOp(return_op);
        return success();
    }
};

//===----------------------------------------------------------------------===//
// LowerToFunc RewritePatterns: Call operations
//===----------------------------------------------------------------------===//

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
        auto funcOp = fsharp::utils::findFunctionInScope(call_op, calleeSymbol.getValue());
        if (!funcOp)
        {
            mlir::emitError(call_op.getLoc(), "Callee function not found in symbol table: ") << calleeSymbol;
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
    // First lower the function calls them
    {
        mlir::ConversionTarget target(getContext());
        target.addLegalDialect<affine::AffineDialect, mlir::BuiltinDialect,
                               arith::ArithDialect, func::FuncDialect,
                               memref::MemRefDialect, fsharp::FSharpDialect>();
        target.addIllegalOp<fsharp::ClosureOp>();
        target.addIllegalOp<fsharp::ReturnOp>();

        RewritePatternSet patterns(&getContext());

        patterns.add<ClosureOpLowering>(&getContext());
        patterns.add<ReturnOpLowering>(&getContext());

        auto module = getOperation();
        if (failed(applyFullConversion(module, target, std::move(patterns))))
            signalPassFailure();
    }

    //mlir::emitError(getOperation().getLoc(), "Function calls lowered to func.call: \n") << *getOperation();

    {
        mlir::ConversionTarget target(getContext());
        target.addLegalDialect<affine::AffineDialect, mlir::BuiltinDialect,
                               arith::ArithDialect, func::FuncDialect,
                               memref::MemRefDialect, fsharp::FSharpDialect>();
        target.addIllegalOp<fsharp::CallOp>();

        RewritePatternSet patterns(&getContext());

        patterns.add<CallOpLowering>(&getContext());

        auto module = getOperation();
        if (failed(applyFullConversion(module, target, std::move(patterns))))
            signalPassFailure();
    }
}

namespace mlir::fsharp
{
    std::unique_ptr<mlir::Pass> createLowerToFunctionPass()
    {
        return std::make_unique<FSharpToFuncLoweringPass>();
    }
}
