//
// Created by lasse on 1/24/25.
//

#include <compiler/FSharpDialect.h>
using namespace mlir;

//===----------------------------------------------------------------------===//
// LowerToArith RewritePatterns: Add operations
//===----------------------------------------------------------------------===//

struct AddOpLowering : public OpConversionPattern<fsharp::AddOp>
{
    using OpConversionPattern<fsharp::AddOp>::OpConversionPattern;

    LogicalResult matchAndRewrite(fsharp::AddOp op, OpAdaptor adaptor,
                                  ConversionPatternRewriter& rewriter) const final
    {
        auto operand_type = op.getOperandTypes()[0];
        if (auto int_type = llvm::dyn_cast_or_null<IntegerType>(operand_type))
        {
            Operation* new_op = rewriter.create<arith::AddIOp>(op.getLoc(), op.getLhs(), op.getRhs());
            op.replaceAllUsesWith(new_op->getResult(0));
            rewriter.eraseOp(op);
            return success();
        }
        else if (auto float_type = llvm::dyn_cast_or_null<FloatType>(operand_type))
        {
            Operation* new_op = rewriter.create<arith::AddFOp>(op.getLoc(), op.getLhs(), op.getRhs());
            op.replaceAllUsesWith(new_op->getResult(0));
            rewriter.eraseOp(op);
            return success();
        }
        return failure();
    }
};

//===----------------------------------------------------------------------===//
// LowerToArith RewritePatterns: SubOp operations
//===----------------------------------------------------------------------===//

struct SubOpLowering : public OpConversionPattern<fsharp::SubOp>
{
    using OpConversionPattern<fsharp::SubOp>::OpConversionPattern;

    LogicalResult matchAndRewrite(fsharp::SubOp op, OpAdaptor adaptor,
                                  ConversionPatternRewriter& rewriter) const final
    {
        auto operand_type = op.getOperandTypes()[0];
        if (auto int_type = llvm::dyn_cast_or_null<IntegerType>(operand_type))
        {
            Operation* new_op = rewriter.create<arith::SubIOp>(op.getLoc(), op.getLhs(), op.getRhs());
            op.replaceAllUsesWith(new_op->getResult(0));
            rewriter.eraseOp(op);
            return success();
        }
        else if (auto float_type = llvm::dyn_cast_or_null<FloatType>(operand_type))
        {
            Operation* new_op = rewriter.create<arith::SubFOp>(op.getLoc(), op.getLhs(), op.getRhs());
            op.replaceAllUsesWith(new_op->getResult(0));
            rewriter.eraseOp(op);
            return success();
        }
        return failure();
    }
};

//===----------------------------------------------------------------------===//
// LowerToArith RewritePatterns: MulOp operations
//===----------------------------------------------------------------------===//

struct MulOpLowering : public OpConversionPattern<fsharp::MulOp>
{
    using OpConversionPattern<fsharp::MulOp>::OpConversionPattern;

    LogicalResult matchAndRewrite(fsharp::MulOp op, OpAdaptor adaptor,
                                  ConversionPatternRewriter& rewriter) const final
    {
        auto operand_type = op.getOperandTypes()[0];
        if (auto int_type = llvm::dyn_cast_or_null<IntegerType>(operand_type))
        {
            Operation* new_op = rewriter.create<arith::MulIOp>(op.getLoc(), op.getLhs(), op.getRhs());
            op.replaceAllUsesWith(new_op->getResult(0));
            rewriter.eraseOp(op);
            return success();
        }
        else if (auto float_type = llvm::dyn_cast_or_null<FloatType>(operand_type))
        {
            Operation* new_op = rewriter.create<arith::MulFOp>(op.getLoc(), op.getLhs(), op.getRhs());
            op.replaceAllUsesWith(new_op->getResult(0));
            rewriter.eraseOp(op);
            return success();
        }
        return failure();
    }
};

//===----------------------------------------------------------------------===//
// LowerToArith RewritePatterns: DivOp operations
//===----------------------------------------------------------------------===//

struct DivOpLowering : public OpConversionPattern<fsharp::DivOp>
{
    using OpConversionPattern<fsharp::DivOp>::OpConversionPattern;

    LogicalResult matchAndRewrite(fsharp::DivOp op, OpAdaptor adaptor,
                                  ConversionPatternRewriter& rewriter) const final
    {
        auto operand_type = op.getOperandTypes()[0];
        if (auto int_type = llvm::dyn_cast_or_null<IntegerType>(operand_type))
        {
            Operation* new_op = nullptr;
            switch (int_type.getSignedness())
            {
            case IntegerType::Signless:
                new_op = rewriter.create<arith::DivSIOp>(op.getLoc(), op.getLhs(), op.getRhs());
                break;
            case IntegerType::Signed:
                new_op = rewriter.create<arith::DivSIOp>(op.getLoc(), op.getLhs(), op.getRhs());
                break;
            case IntegerType::Unsigned:
                new_op = rewriter.create<arith::DivUIOp>(op.getLoc(), op.getLhs(), op.getRhs());
                break;
            }

            op.replaceAllUsesWith(new_op->getResult(0));
            rewriter.eraseOp(op);
            return success();
        }
        else if (auto float_type = llvm::dyn_cast_or_null<FloatType>(operand_type))
        {
            Operation* new_op = rewriter.create<arith::DivFOp>(op.getLoc(), op.getLhs(), op.getRhs());
            op.replaceAllUsesWith(new_op->getResult(0));
            rewriter.eraseOp(op);
            return success();
        }
        return failure();
    }
};

//===----------------------------------------------------------------------===//
// LowerToArith RewritePatterns: ModOp operations
//===----------------------------------------------------------------------===//

struct ModOpLowering : public OpConversionPattern<fsharp::ModOp>
{
    using OpConversionPattern<fsharp::ModOp>::OpConversionPattern;

    LogicalResult matchAndRewrite(fsharp::ModOp op, OpAdaptor adaptor,
                                  ConversionPatternRewriter& rewriter) const final
    {
        auto operand_type = op.getOperandTypes()[0];
        if (auto int_type = llvm::dyn_cast_or_null<IntegerType>(operand_type))
        {
            Operation* new_op = nullptr;
            switch (int_type.getSignedness())
            {
            case IntegerType::Signless:
                new_op = rewriter.create<arith::RemSIOp>(op.getLoc(), op.getLhs(), op.getRhs());
                break;
            case IntegerType::Signed:
                new_op = rewriter.create<arith::RemSIOp>(op.getLoc(), op.getLhs(), op.getRhs());
                break;
            case IntegerType::Unsigned:
                new_op = rewriter.create<arith::RemUIOp>(op.getLoc(), op.getLhs(), op.getRhs());
                break;
            }

            op.replaceAllUsesWith(new_op->getResult(0));
            rewriter.eraseOp(op);
            return success();
        }
        else if (auto float_type = llvm::dyn_cast_or_null<FloatType>(operand_type))
        {
            Operation* new_op = rewriter.create<arith::RemFOp>(op.getLoc(), op.getLhs(), op.getRhs());
            op.replaceAllUsesWith(new_op->getResult(0));
            rewriter.eraseOp(op);
            return success();
        }
        return failure();
    }
};

//===----------------------------------------------------------------------===//
// LowerToArith RewritePatterns: DivOp operations
//===----------------------------------------------------------------------===//

struct ConstantOpLowering : public OpConversionPattern<fsharp::ConstantOp>
{
    using OpConversionPattern<fsharp::ConstantOp>::OpConversionPattern;

    LogicalResult matchAndRewrite(fsharp::ConstantOp op, OpAdaptor adaptor,
                                  ConversionPatternRewriter& rewriter) const final
    {
        if (auto int_attr = mlir::dyn_cast_or_null<mlir::IntegerAttr>(op.getValue()))
        {
            auto type = mlir::dyn_cast<IntegerType>(int_attr.getType());
            Operation* new_op = rewriter.create<arith::ConstantOp>(op.getLoc(),
                                                                   IntegerAttr::get(
                                                                       IntegerType::get(
                                                                           op.getContext(), type.getWidth()),
                                                                       int_attr.getValue()));;
            op.replaceAllUsesWith(new_op->getResult(0));
            rewriter.eraseOp(op);
            return success();
        }
        else if (auto float_attr = mlir::dyn_cast_or_null<mlir::FloatAttr>(op.getValue()))
        {
            Operation* new_op = rewriter.create<arith::ConstantOp>(op.getLoc(),
                                                                   FloatAttr::get(
                                                                       FloatType::getF64(op.getContext()),
                                                                       float_attr.getValue()));
            op.replaceAllUsesWith(new_op->getResult(0));
            rewriter.eraseOp(op);
            return success();
        }
        else if (auto dense_attr = mlir::dyn_cast_or_null<mlir::DenseElementsAttr>(op.getValue()))
        {
            Operation* new_op = rewriter.create<arith::ConstantOp>(op.getLoc(), op.getValue());
            op.replaceAllUsesWith(new_op->getResult(0));
            rewriter.eraseOp(op);
            return success();
        }
        return failure();
    }
};

//===----------------------------------------------------------------------===//
// TypeInferencePass
//===----------------------------------------------------------------------===//

namespace
{
    struct LowerToArithPass
        : public PassWrapper<LowerToArithPass, OperationPass<ModuleOp>>
    {
        MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LowerToArithPass)

        void getDependentDialects(DialectRegistry& registry) const override
        {
            registry.insert<func::FuncDialect>();
        }

        void convertToSignless(ModuleOp module)
        {
            module.walk([](Operation* op)
            {
                // Iterate over operands
                for (Value operand : op->getOperands())
                {
                    if (auto intType = dyn_cast<IntegerType>(operand.getType()))
                    {
                        if (intType.isSigned() || intType.isUnsigned())
                        {
                            // Replace with signless integer type
                            operand.setType(IntegerType::get(op->getContext(), intType.getWidth()));
                        }
                    }
                }

                // Iterate over results
                for (Value result : op->getResults())
                {
                    if (auto intType = dyn_cast<IntegerType>(result.getType()))
                    {
                        if (intType.isSigned() || intType.isUnsigned())
                        {
                            // Replace with signless integer type
                            result.setType(IntegerType::get(op->getContext(), intType.getWidth()));
                        }
                    }
                }

                if (auto function_op = dyn_cast<FunctionOpInterface>(op))
                {
                    FunctionType f_type = cast<FunctionType>(function_op.getFunctionType());

                    SmallVector<Type, 4> newInputs(f_type.getInputs());
                    SmallVector<Type, 4> newOutputs(f_type.getResults());

                    // Iterate over operands
                    for (auto [i, operand_type] : llvm::enumerate(newInputs))
                    {
                        if (auto intType = dyn_cast<IntegerType>(operand_type))
                        {
                            if (intType.isSigned() || intType.isUnsigned())
                            {
                                // Replace with signless integer type
                                newInputs[i] = IntegerType::get(op->getContext(), intType.getWidth());
                            }
                        }
                    }

                    // Iterate over results
                    for (auto [i, result_type] : llvm::enumerate(newOutputs))
                    {
                        if (auto intType = dyn_cast<IntegerType>(result_type))
                        {
                            if (intType.isSigned() || intType.isUnsigned())
                            {
                                // Replace with signless integer type
                                newOutputs[i] = IntegerType::get(op->getContext(), intType.getWidth());
                            }
                        }
                    }
                    function_op.setType(FunctionType::get(function_op.getContext(), newInputs, newOutputs));
                }
            });
        }

        void runOnOperation() final
        {
            // Lower all the arithmetic operations
            {
                mlir::ConversionTarget target(getContext());
                target.addLegalDialect<affine::AffineDialect, mlir::BuiltinDialect,
                                       arith::ArithDialect, func::FuncDialect,
                                       memref::MemRefDialect>();
                target.addIllegalDialect<fsharp::FSharpDialect>();
                target.addLegalOp<fsharp::ClosureOp>();
                target.addLegalOp<fsharp::ReturnOp>();
                target.addLegalOp<fsharp::CallOp>();
                target.addLegalOp<fsharp::PrintOp>();
                target.addLegalOp<fsharp::ConstantOp>();

                RewritePatternSet patterns(&getContext());

                // We only want to lower the fsharp arithmetic operations
                patterns.add<AddOpLowering>(&getContext());
                patterns.add<SubOpLowering>(&getContext());
                patterns.add<MulOpLowering>(&getContext());
                patterns.add<DivOpLowering>(&getContext());
                patterns.add<ModOpLowering>(&getContext());

                auto module = getOperation();
                if (failed(applyFullConversion(module, target, std::move(patterns))))
                    signalPassFailure();
            }

            // Now lower all the constant ops
            {
                mlir::ConversionTarget target(getContext());
                target.addLegalDialect<affine::AffineDialect, mlir::BuiltinDialect,
                                       arith::ArithDialect, func::FuncDialect,
                                       memref::MemRefDialect>();
                target.addIllegalDialect<fsharp::FSharpDialect>();
                target.addLegalOp<fsharp::ClosureOp>();
                target.addLegalOp<fsharp::ReturnOp>();
                target.addLegalOp<fsharp::CallOp>();
                target.addLegalOp<fsharp::PrintOp>();

                RewritePatternSet patterns(&getContext());

                // We only want to lower the fsharp constant operations
                patterns.add<ConstantOpLowering>(&getContext());

                auto module = getOperation();
                if (failed(applyFullConversion(module, target, std::move(patterns))))
                    signalPassFailure();
            }

            convertToSignless(getOperation());
        }
    };
} // namespace


namespace mlir::fsharp
{
    std::unique_ptr<mlir::Pass> createLowerToArithPass()
    {
        return std::make_unique<LowerToArithPass>();
    }
}
