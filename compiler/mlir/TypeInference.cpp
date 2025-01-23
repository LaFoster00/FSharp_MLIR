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

        void runOnOperation() final
        {
            auto f = getOperation();

            // Infer the operations with all inferred operands first
            {
                // Populate the worklist with the operations that need shape inference:
                // these are operations that return a dynamic shape.
                fsharp::ClosureOp bla;
                llvm::SmallPtrSet<mlir::Operation*, 16> unknown_return_work_list;
                f.walk([&](mlir::Operation* op)
                {
                    if (fsharp::utils::returnsUnknownType(op))
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
                        shapeOp.inferFromOperands();
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
