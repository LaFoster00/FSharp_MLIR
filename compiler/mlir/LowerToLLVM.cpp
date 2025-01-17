//
// Created by lasse on 1/14/25.
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
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/Support/Casting.h"
#include <memory>
#include <utility>
#include <mlir/Dialect/LLVMIR/FunctionCallUtils.h>

using namespace mlir;

static LLVM::LLVMFunctionType getPrintfType(MLIRContext* context)
{
    return LLVM::LLVMFunctionType::get(LLVM::LLVMVoidType::get(context),
                                       LLVM::LLVMPointerType::get(context),
                                       true);
}

/// Return a value representing an access into a global string with the given
/// name, creating the string if necessary.
static Value getOrCreateGlobalString(Location loc, OpBuilder& builder,
                                     StringRef name, StringRef value)
{
    assert(builder.getInsertionBlock() &&
        builder.getInsertionBlock()->getParentOp() &&
        "expected builder to point to a block constrained in an op");
    auto module =
        builder.getInsertionBlock()->getParentOp()->getParentOfType<ModuleOp>();
    assert(module && "builder points to an op outside of a module");

    LLVM::GlobalOp global;
    MLIRContext* ctx = builder.getContext();
    auto type = LLVM::LLVMArrayType::get(IntegerType::get(ctx, 8), value.size());
    // Only create string if it doesn't exist already.
    if (!(global = module.lookupSymbol<LLVM::GlobalOp>(name)))
    {
        OpBuilder::InsertionGuard insertGuard(builder);
        // Create the global at the entry of the module.
        OpBuilder moduleBuilder(module.getBodyRegion(), builder.getListener());
        global = moduleBuilder.create<LLVM::GlobalOp>(
            loc, type, /*isConstant=*/true, LLVM::Linkage::Internal, name,
            builder.getStringAttr(value), /*alignment=*/0);
    }

    LLVM::LLVMPointerType ptrType = LLVM::LLVMPointerType::get(ctx);
    // Get the pointer to the first character in the global string.
    Value globalPtr =
        builder.create<LLVM::AddressOfOp>(loc, ptrType, global.getSymNameAttr());
    return builder.create<LLVM::GEPOp>(loc, ptrType, type, globalPtr,
                                       ArrayRef<LLVM::GEPArg>{0, 0});
}

namespace
{
    /// Lowers `fsharp.print` to either a raw call to printf or a loop nest calling `printf` on each of the individual
    /// elements of the array. // TODO implement array support
    class PrintOpLowering : public mlir::ConversionPattern
    {
    public:
        explicit PrintOpLowering(MLIRContext* context)
            : ConversionPattern(fsharp::PrintOp::getOperationName(), 1, context)
        {
        }

        LogicalResult matchAndRewrite(Operation* op, ArrayRef<Value> operands,
                                      ConversionPatternRewriter& rewriter) const override
        {
            auto* context = rewriter.getContext();

            auto loc = op->getLoc();

            ModuleOp parentModule = op->getParentOfType<ModuleOp>();

            // Get a symbol reference to the printf function, inserting it if necessary.
            auto printfRef = LLVM::lookupOrCreateFn(parentModule,
                                                    "printf",
                                                    getPrintfType(context).getParams(),
                                                    getPrintfType(context).getReturnType(), true);

            for (auto operand_type = std::next(op->operand_type_begin()); operand_type != op->operand_type_end(); ++
                 operand_type)
            {
                if (mlir::isa<mlir::ShapedType>(*operand_type) && mlir::dyn_cast<ShapedType>(*operand_type).
                    getElementType() != mlir::IntegerType::get(context, 8))
                {
                    mlir::emitError(loc, "Array support not implemented yet");
                    return failure();
#if 0
                    auto memRefType = mlir::cast<MemRefType>(*operand_type);
                    auto memRefShape = memRefType.getShape();

                    Value formatSpecifierCst = getOrCreateGlobalString(
                        loc, rewriter, "frmt_spec", StringRef("%f \0", 4));
                    Value newLineCst = getOrCreateGlobalString(
                        loc, rewriter, "nl", StringRef("\n\0", 2));

                    // Create a loop for each of the dimensions within the shape.
                    SmallVector<Value, 4> loopIvs;
                    for (unsigned i = 0, e = memRefShape.size(); i != e; ++i)
                    {
                        auto lowerBound = rewriter.create<arith::ConstantIndexOp>(loc, 0);
                        auto upperBound =
                            rewriter.create<arith::ConstantIndexOp>(loc, memRefShape[i]);
                        auto step = rewriter.create<arith::ConstantIndexOp>(loc, 1);
                        auto loop =
                            rewriter.create<scf::ForOp>(loc, lowerBound, upperBound, step);
                        for (Operation& nested : *loop.getBody())
                            rewriter.eraseOp(&nested);
                        loopIvs.push_back(loop.getInductionVar());

                        // Terminate the loop body.
                        rewriter.setInsertionPointToEnd(loop.getBody());

                        // Insert a newline after each of the inner dimensions of the shape.
                        if (i != e - 1)
                            rewriter.create<LLVM::CallOp>(loc, printfRef, newLineCst);
                        rewriter.create<scf::YieldOp>(loc);
                        rewriter.setInsertionPointToStart(loop.getBody());
                    }

                    // Generate a call to printf for the current element of the loop.
                    auto printOp = cast<fsharp::PrintOp>(op);
                    auto elementLoad =
                        rewriter.create<memref::LoadOp>(loc, printOp.getOperand(0), loopIvs);
                    rewriter.create<LLVM::CallOp>(
                        loc, printfRef,
                        ArrayRef<Value>({formatSpecifierCst, elementLoad}));

                    // Notify the rewriter that this operation has been removed.
                    rewriter.eraseOp(op);
#endif
                }
            }

            auto printOp = mlir::dyn_cast<fsharp::PrintOp>(op);
            mlir::SmallVector<mlir::Value, 4> updated_operands;
            for (auto operand : op->getOperands())
            {
                // If the oprand isn't a shaped type, we can just add it to the updated operands.
                if (!operand.getType().isa<mlir::ShapedType>())
                {
                    updated_operands.push_back(operand);
                    continue;
                }
                auto pointer_index = rewriter.create<mlir::memref::ExtractAlignedPointerAsIndexOp>(loc, operand);
                auto arith_index = rewriter.create<mlir::arith::IndexCastOp>(
                    loc, IntegerType::get(context, 64), pointer_index);
                auto llvm_pointer = rewriter.create<LLVM::IntToPtrOp>(loc, LLVM::LLVMPointerType::get(context),
                                                                      arith_index);
                updated_operands.push_back(llvm_pointer);
            }
            rewriter.create<LLVM::CallOp>(loc, printfRef, updated_operands);

            // Notify the rewriter that this operation has been removed.
            rewriter.eraseOp(op);
            return success();
        }
    };
} // namespace

//===----------------------------------------------------------------------===//
// FSharpToLLVMLoweringPass
//===----------------------------------------------------------------------===//

namespace
{
    struct FSharpToLLVMLoweringPass
        : public PassWrapper<FSharpToLLVMLoweringPass, OperationPass<ModuleOp>>
    {
        MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(FSharpToLLVMLoweringPass)

        void getDependentDialects(DialectRegistry& registry) const override
        {
            registry.insert<LLVM::LLVMDialect, scf::SCFDialect>();
        }

        void runOnOperation() final;
    };
} // namespace

void FSharpToLLVMLoweringPass::runOnOperation()
{
    // The first thing to define is the conversion target. This will define the
    // final target for this lowering. For this lowering, we are only targeting
    // the LLVM dialect.
    mlir::LLVMConversionTarget target(getContext());
    target.addLegalOp<ModuleOp>();

    // During this lowering, we will also be lowering the MemRef types, that are
    // currently being operated on, to a representation in LLVM. To perform this
    // conversion we use a TypeConverter as part of the lowering. This converter
    // details how one type maps to another. This is necessary now that we will be
    // doing more complicated lowerings, involving loop region arguments.
    mlir::LLVMTypeConverter typeConverter(&getContext());

    // Now that the conversion target has been defined, we need to provide the
    // patterns used for lowering. At this point of the compilation process, we
    // have a combination of `toy`, `affine`, and `std` operations. Luckily, there
    // are already exists a set of patterns to transform `affine` and `std`
    // dialects. These patterns lowering in multiple stages, relying on transitive
    // lowerings. Transitive lowering, or A->B->C lowering, is when multiple
    // patterns must be applied to fully transform an illegal operation into a
    // set of legal ones.
    RewritePatternSet patterns(&getContext());
    populateAffineToStdConversionPatterns(patterns);
    populateSCFToControlFlowConversionPatterns(patterns);
    mlir::arith::populateArithToLLVMConversionPatterns(typeConverter, patterns);
    populateFinalizeMemRefToLLVMConversionPatterns(typeConverter, patterns);
    cf::populateControlFlowToLLVMConversionPatterns(typeConverter, patterns);
    populateFuncToLLVMConversionPatterns(typeConverter, patterns);

    // The only remaining operation to lower from the `toy` dialect, is the
    // PrintOp.
    patterns.add<PrintOpLowering>(&getContext());

    // We want to completely lower to LLVM, so we use a `FullConversion`. This
    // ensures that only legal operations will remain after the conversion.
    auto module = getOperation();
    if (failed(applyFullConversion(module, target, std::move(patterns))))
        signalPassFailure();
}

/// Create a pass for lowering operations the remaining `Toy` operations, as
/// well as `Affine` and `Std`, to the LLVM dialect for codegen.
std::unique_ptr<mlir::Pass> mlir::fsharp::createLowerToLLVMPass()
{
    return std::make_unique<FSharpToLLVMLoweringPass>();
}
