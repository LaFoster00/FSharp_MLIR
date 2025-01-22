//
// Created by lasse on 22/01/2025.
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
// TypeInferencePass
//===----------------------------------------------------------------------===//

namespace
{
    struct TypeInferencePass
        : public PassWrapper<TypeInferencePass, OperationPass<ModuleOp>>
    {
        MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TypeInferencePass)

        void getDependentDialects(DialectRegistry& registry) const override
        {
            registry.insert<affine::AffineDialect, func::FuncDialect,
                            memref::MemRefDialect>();
        }

        void runOnOperation() final
        {
            auto f = getOperation();

            // Populate the worklist with the operations that need shape inference:
            // these are operations that return a dynamic shape.
            llvm::SmallPtrSet<mlir::Operation *, 16> opWorklist;
            f.walk([&](mlir::Operation *op) {
              if (returnsUnknownType(op))
                  opWorklist.insert(op);
              });

            // Iterate on the operations in the worklist until all operations have been
            // inferred or no change happened (fix point).
            while (!opWorklist.empty())
            {
                // Find the next operation ready for inference, that is an operation
                // with all operands already resolved (non-generic).
                auto nextop = llvm::find_if(opWorklist, allOperandsInferred);
                if (nextop == opWorklist.end())
                    break;

                Operation *op = *nextop;
                opWorklist.erase(op);

                // Ask the operation to infer its output shapes.
                llvm::dbgs() << "Inferring shape for: " << *op << "\n";
                if (auto shapeOp = dyn_cast<TypeInference>(op))
                {
                    shapeOp.inferTypes();
                } else
                {
                    op->emitError("Unable to infer type of operation without type inference interface. \n"
                                  "This op is likely not compatible with type inference. All types should be resolved prior"
                                  "to this op.");
                    return signalPassFailure();
                }
            }

            // In case all possible types have been inferred we can now assume that the remaining types should be integers.
            while (!opWorklist.empty())
            {
                // Find the next operation ready for inference, that is an operation
                // with all operands already resolved (non-generic).
                auto nextop = llvm::find_if(opWorklist, allOperandsInferred);
                if (nextop == opWorklist.end())
                    break;

                Operation *op = *nextop;
                opWorklist.erase(op);

                // Ask the operation to infer its output shapes.
                llvm::dbgs() << "Inferring shape for: " << *op << "\n";
                if (auto shapeOp = dyn_cast<TypeInference>(op))
                {
                    shapeOp.inferTypes();
                } else
                {
                    op->emitError("Unable to infer type of operation without type inference interface. \n"
                                  "This op is likely not compatible with type inference. All types should be resolved prior"
                                  "to this op.");
                    return signalPassFailure();
                }
            }
        }

        /// A utility method that returns if the given operation has all of its
        /// operands inferred.
        static bool allOperandsInferred(Operation *op) {
            return llvm::all_of(op->getOperandTypes(), [](Type operandType) {
              return !llvm::isa<NoneType>(operandType);
            });
        }

        /// A utility method that returns if the given operation has a dynamically
        /// shaped result.
        static bool returnsUnknownType(Operation *op) {
            return llvm::any_of(op->getResultTypes(), [](Type resultType) {
              return llvm::isa<NoneType>(resultType);
            });
        }
    };
} // namespace


namespace mlir::fsharp
{
    std::unique_ptr<mlir::Pass> createTypeInferencePass()
    {
        return std::make_unique<TypeInferencePass>();
    }
}
