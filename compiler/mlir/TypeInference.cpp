//
// Created by lasse on 22/01/2025.
//

#include <compiler/CompilerUtils.h>

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

        static uint32_t opsToResolve(ModuleOp module_op)
        {
            uint32_t open_ops = 0;
            module_op.walk([&](mlir::Operation* op)
            {
                if (!fsharp::utils::allOperandsInferred(op) || fsharp::utils::returnsUnknownType(op))
                {
                    ++open_ops;
                }
            });
            return open_ops;
        }

        // Infers all operations from their operand types if possible.
        void inferFromOperands(ModuleOp module_op)
        {
            // Populate the worklist with the operations that need shape inference:
            // these are operations that return a dynamic shape.
            llvm::SmallPtrSet<mlir::Operation*, 16> unknown_return_work_list;
            module_op.walk([&](mlir::Operation* op)
            {
                if (!fsharp::utils::isImplicitTypeInferred(op) && fsharp::utils::returnsUnknownType(op))
                    unknown_return_work_list.insert(op);
            });

            // Infer all the types that we can from the defined operands.
            while (!unknown_return_work_list.empty())
            {
                // Find the next operation ready for inference, that is an operation
                // with all operands already resolved (non-generic).
                auto nextop = llvm::find_if(unknown_return_work_list, fsharp::utils::allOperandsInferred);
                if (nextop == unknown_return_work_list.end())
                    break;

                Operation* op = *nextop;
                unknown_return_work_list.erase(op);
                // Ask the operation to infer its output shapes.
                if (auto shapeOp = dyn_cast<TypeInference>(op))
                {
                    //shapeOp->emitError("Inferring: ") << shapeOp;
                    shapeOp.inferFromOperands();
                    //shapeOp->emitError("Inferred: ") << shapeOp << '\n';
                }
                else
                {
                    op->emitError("Unable to infer type of operation without type inference interface. \n"
                        "This op is likely not compatible with type inference. All types should be resolved prior"
                        "to this op.") << op;
                    return signalPassFailure();
                }
            }

            //mlir::emitError(module_op.getLoc(), "Inferred from operands: \n") << *module_op;
        }

        void inferFromReturnType(ModuleOp module_op)
        {
            // Populate the worklist with the operations that need shape inference:
            // these are operations that return a dynamic shape.
            llvm::SmallPtrSet<mlir::Operation*, 16> unknown_return_work_list;
            module_op.walk([&](mlir::Operation* op)
            {
                if (!fsharp::utils::isImplicitTypeInferred(op)
                    && !fsharp::utils::allOperandsInferred(op)
                    && fsharp::utils::returnsKnownType(op))
                {
                    unknown_return_work_list.insert(op);
                    //mlir::emitError(op->getLoc(), "Inferring from return type: ") << op;
                }
            });

            // Infer all the types that we can from the defined operands.
            while (!unknown_return_work_list.empty())
            {
                // Find the next operation ready for inference, that is an operation
                // with all operands already resolved (non-generic).
                auto nextop = llvm::find_if(unknown_return_work_list, fsharp::utils::returnsKnownType);
                if (nextop == unknown_return_work_list.end())
                    break;

                Operation* op = *nextop;
                unknown_return_work_list.erase(op);
                // Ask the operation to infer its output shapes.
                if (auto shapeOp = dyn_cast<TypeInference>(op))
                {
                    shapeOp.inferFromReturnType();
                }
                else
                {
                    op->emitError("Unable to infer type of operation without type inference interface. \n"
                        "This op is likely not compatible with type inference. All types should be resolved prior"
                        "to this op.") << op;
                    return signalPassFailure();
                }
            }

            //mlir::emitError(module_op.getLoc(), "Inferred from return type: \n") << *module_op;
        }

        // Infers all operations that have a specific way of resolving unknown types.
        void inferFromUnknown(ModuleOp module_op)
        {
            llvm::SmallVector<Operation*, 16> work_list;
            module_op.walk([&](mlir::Operation* op)
            {
               if (fsharp::utils::noOperandsInferred(op) && fsharp::utils::returnsUnknownType(op) && !fsharp::utils::isImplicitTypeInferred(op))
               {
                   work_list.push_back(op);
               }
            });

            for (auto op : work_list)
            {
                if (auto shapeOp = dyn_cast<TypeInference>(op))
                {
                    shapeOp.inferFromUnknown();
                }
            }

            //mlir::emitError(module_op.getLoc(), "Inferred from unkown: \n") << *module_op;
        }

        void runOnOperation() final
        {
            auto f = getOperation();
            // Continue resolving types as long as we are making progress.
            uint32_t last_open_ops = 0;
            uint32_t current_open_ops = opsToResolve(f);
            while (last_open_ops != current_open_ops)
            {
                last_open_ops = current_open_ops;
                inferFromReturnType(f);
                inferFromOperands(f);
                current_open_ops = opsToResolve(f);
            }
            // At last assume int type for all unresolved types.
            inferFromUnknown(f);

            // Go over the operations again and resolve any remaining types from the operations that were resolved
            // Anything that cant be resolved during this step is a generic which is not supported and therefore error.
            current_open_ops = opsToResolve(f);
            while (last_open_ops != current_open_ops)
            {
                last_open_ops = current_open_ops;
                inferFromReturnType(f);
                inferFromOperands(f);
                current_open_ops = opsToResolve(f);
            }

            if (current_open_ops != 0)
            {
                mlir::emitError(getOperation().getLoc(), "Unable to resolve all types in the module. \n"
                    "This is likely due to a generic operation that is not supported by the type inference pass. \n"
                    "All types should be resolved prior to this pass.") << *getOperation();
                return signalPassFailure();
            }

            // TODO Update the formating parameters for the print statements so that they are c compatible
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
