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
                else if (!op->hasTrait<OpTrait::IsTerminator>()
                    && !op->hasTrait<InferTypeOpInterface::Trait>())
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
                else if (!op->hasTrait<OpTrait::IsTerminator>()
                    && !op->hasTrait<InferTypeOpInterface::Trait>())
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
                if (!fsharp::utils::allOperandsInferred(op)
                    && fsharp::utils::returnsUnknownType(op)
                    && !fsharp::utils::isImplicitTypeInferred(op))
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

        // Resolve the print formatting strings to be compatible with formating specifiers. This is necessary since some
        // of the formatting specifiers are not compatible with the C standard (mostly length of integers and signedness).
        void resolvePrintFormatingStringsToC(fsharp::PrintOp print_op)
        {
            auto format_types = fsharp::utils::getFormatSpecifiedTypes(
                print_op.getFmtString(),
                print_op.getContext()
            );

            std::string new_format = print_op.getFmtString().str();

            auto operands = print_op.getFmtOperands();
            // Define a simple state machine to parse the format string
            int i = 0;
            int operand_i = 0;
            while (i < new_format.size())
            {
                if (new_format[i] == '%')
                {
                    ++i; // Advance to the character after '%'

                    // Skip flags and width/precision modifiers (e.g., "%-10.3d").
                    while (i < new_format.size() && (new_format[i] == '-' || new_format[i] == '+' ||
                        new_format[i] == ' ' || new_format[i] == '#' ||
                        new_format[i] == '0' || std::isdigit(new_format[i]) ||
                        new_format[i] == '.'))
                    {
                        ++i;
                    }

                    // Ensure we haven't reached the end of the string.
                    if (i >= new_format.size())
                    {
                        break;
                    }

                    // Check the specifier and map to an MLIR type.
                    switch (new_format[i])
                    {
                    case 'b':
                        new_format[i] = 'u'; // Bool is unsigned
                        break;
                    case 'i': // Integer
                    case 'd':
                        if (auto int_type = mlir::dyn_cast<IntegerType>(operands[operand_i++].getType()))
                        {
                            if (int_type.getWidth() == 8)
                            {
                                new_format[i] = 'd'; // Treat char as signed int
                            }
                            else if (int_type.getWidth() == 32)
                            {
                                if (int_type.isSigned())
                                    new_format[i] = 'i'; // 32 bit signed integer
                                else
                                    new_format[i] = 'u'; // 32 bit unsigned integer
                            }
                            else
                            {
                                if (int_type.isSigned())
                                {
                                    // 64 bit unsigned integer
                                    new_format[i] = 'i';
                                    // Insert l at position of the i, results in %lli
                                    new_format.insert(i, "ll");
                                }
                                else
                                {
                                    // 64 bit unsigned integer
                                    new_format[i] = 'u';
                                    // Insert l at position of the u, results in %llu
                                    new_format.insert(i, "ll");
                                }
                            }
                        }
                    default:
                        // Leave the format specifier as is
                        break;
                    }
                }
                else
                {
                    ++i; // Advance to the next character
                }
            }

            print_op.setFmtString(new_format);
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

            f.walk([&](fsharp::PrintOp print_op)
            {
                resolvePrintFormatingStringsToC(print_op);
            });
        }
    };
} // namespace


namespace
mlir::fsharp
{
    std::unique_ptr<mlir::Pass> createTypeInferencePass()
    {
        return std::make_unique<TypeInferencePass>();
    }
}
