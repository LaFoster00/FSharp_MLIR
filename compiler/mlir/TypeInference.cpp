//
// Created by lasse on 22/01/2025.
//

#include "compiler/FSharpDialect.h"
#include "compiler/FSharpPasses.h"

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
            llvm::SmallPtrSet<mlir::Operation*, 16> opWorklist;
            f.walk([&](mlir::Operation* op)
            {
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

                Operation* op = *nextop;
                // Ask the operation to infer its output shapes.
                if (auto shapeOp = dyn_cast<TypeInference>(op))
                {
                    switch (int result = shapeOp.inferTypes())
                    {
                    case 0: // No types were resolved
                    case 1: // Some types were resolved
                        break;
                    case 2: // All types were resolved
                        opWorklist.erase(op);
                        break;
                    }
                }
                else
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

                Operation* op = *nextop;

                // Ask the operation to infer its output shapes.
                if (auto shapeOp = dyn_cast<TypeInference>(op))
                {
                    shapeOp.assumeTypes();
                }
                else
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
        static bool allOperandsInferred(Operation* op)
        {
            return llvm::all_of(op->getOperandTypes(), [](Type operandType)
            {
                return !llvm::isa<NoneType>(operandType);
            });
        }

        /// A utility method that returns if the given operation has a dynamically
        /// shaped result.
        static bool returnsUnknownType(Operation* op)
        {
            return llvm::any_of(op->getResultTypes(), [](Type resultType)
            {
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
