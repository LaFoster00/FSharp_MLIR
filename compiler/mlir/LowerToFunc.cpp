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
                // This is done by getting the region the operand belongs to and checking if it is the same as the closure region
                // or if the closure_op is a parent to the operand parent region.
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
                                                     closure_op->getParentOfType<ModuleOp>())))
        {
            return failure();
        }
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
// LowerToFunc RewritePatterns: AssertOp operations
//===----------------------------------------------------------------------===//

GENERATE_OP_CONVERSION_PATTERN(AssertOp)
    rewriter.replaceOpWithNewOp<cf::AssertOp>(op, op.getCondition(), op.getMessageAttr());
    return success();
END_GENERATE_OP_CONVERSION_PATTERN()


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
    // Partially lower the print ops here as well
    {
        // We need to lower the print operation partially and extract the format string into a tensor operand so that
        // the bufferization pass handles it correctly. Without a tensor operand, the bufferization pass will not run for the print operation.
        auto module = getOperation();
        module->walk([&](fsharp::PrintOp op)
        {
            OpBuilder builder(op);

            SmallVector<Value> newOperands;

            auto fmtString = op.getFmtStringAttr();

            // Copy the string data into a tensor.
            std::vector<char8_t> data{fmtString.begin(), fmtString.end()};
            data.push_back('\0');
            auto type = RankedTensorType::get({static_cast<int64_t>(data.size())}, builder.getI8Type());
            auto dataAttribute = DenseElementsAttr::get(type, llvm::ArrayRef(data));
            auto tensor = builder.create<arith::ConstantOp>(op.getLoc(), type, dataAttribute);
            // Add the string tensor as the first operand.
            newOperands.push_back(tensor);
            // Add the rest of the operands.
            auto oldOperands = op.getFmtOperands();
            newOperands.append(oldOperands.begin(), oldOperands.end());
            //Create the new CallOp.
            auto newPrintOp = builder.create<fsharp::PrintOp>(op.getLoc(), op.getFmtStringAttr(), newOperands);
            newPrintOp.getOperation()->setAttrs(op->getAttrs());
            fsharp::utils::addOrUpdateAttrDictEntry(newPrintOp, "analysisFinished", builder.getUnitAttr());

            op->replaceAllUsesWith(newPrintOp);
            op.erase();
        });
    }

    // Lower the functions to func dialect operations
    {
        mlir::ConversionTarget target(getContext());
        target.addLegalDialect<affine::AffineDialect,
                               mlir::BuiltinDialect,
                               arith::ArithDialect,
                               func::FuncDialect,
                               scf::SCFDialect,
                               memref::MemRefDialect,
                               tensor::TensorDialect,
                               cf::ControlFlowDialect>();

        // Make the whole fsharp dialect illegal. Except for the CallOps which will be lowered to func.call in the next step.
        target.addIllegalDialect<fsharp::FSharpDialect>();
        target.addLegalOp<fsharp::CallOp>();
        target.addLegalOp<fsharp::PrintOp>();
        target.addLegalOp<fsharp::AssertOp>();

        RewritePatternSet patterns(&getContext());

        patterns.add<ClosureOpLowering>(&getContext());
        patterns.add<ReturnOpLowering>(&getContext());

        auto module = getOperation();
        if (failed(applyFullConversion(module, target, std::move(patterns))))
            return signalPassFailure();
    }

    //mlir::emitError(getOperation().getLoc(), "Function calls lowered to func.call: \n") << *getOperation();

    // Lower the fsharp.call to func.call and the fsharp.assert to cf.assert operations
    {
        mlir::ConversionTarget target(getContext());
        target.addLegalDialect<affine::AffineDialect,
                               mlir::BuiltinDialect,
                               arith::ArithDialect,
                               func::FuncDialect,
                               scf::SCFDialect,
                               memref::MemRefDialect,
                               tensor::TensorDialect,
                               cf::ControlFlowDialect>();
        target.addIllegalDialect<fsharp::FSharpDialect>();
        target.addLegalOp<fsharp::PrintOp>();

        RewritePatternSet patterns(&getContext());

        patterns.add<CallOpLowering>(&getContext());
        patterns.add<AssertOpLowering>(&getContext());


        auto module = getOperation();
        if (failed(applyFullConversion(module, target, std::move(patterns))))
            return signalPassFailure();
    }
}

namespace mlir::fsharp
{
    std::unique_ptr<mlir::Pass> createLowerToFunctionPass()
    {
        return std::make_unique<FSharpToFuncLoweringPass>();
    }
}
