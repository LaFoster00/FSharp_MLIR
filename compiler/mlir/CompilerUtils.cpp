//
// Created by lasse on 1/23/25.
//
#include "compiler/CompilerUtils.h"

namespace mlir::fsharp::utils
{
    mlir::fsharp::ClosureOp findClosureInScope(mlir::Operation* startOp, mlir::StringRef closureName)
    {
        mlir::Operation* currentOp = startOp;

        // Traverse up through parent operations (or regions) to find the closure
        while (currentOp)
        {
            // Check if the current operation has a SymbolTable
            if (currentOp->hasTrait<mlir::OpTrait::SymbolTable>())
            {
                // Try to lookup the closure in the current SymbolTable
                mlir::Operation* closure = mlir::SymbolTable::lookupSymbolIn(currentOp, closureName);
                if (auto closure_op = mlir::dyn_cast<mlir::fsharp::ClosureOp>(closure))
                {
                    return closure_op; // Found the closure
                }
            }

            // Move to the parent operation
            currentOp = currentOp->getParentOp();
        }

        // If no closure was found, return nullptr
        return nullptr;
    }

    bool someOperandsInferred(Operation* op)
    {
        return llvm::any_of(op->getOperandTypes(), [](Type operandType)
        {
            return !llvm::isa<NoneType>(operandType);
        });
    }


    bool allOperandsInferred(Operation* op)
    {
        ;
        if (op->hasTrait<FunctionOpInterface::Trait>())
        {
            auto function = mlir::dyn_cast_or_null<FunctionOpInterface>(op);
            return llvm::all_of(function.getArgumentTypes(), [](Type operandType)
            {
                return !llvm::isa<NoneType>(operandType);
            });
        }
        else if (auto returnOp = mlir::dyn_cast_or_null<ReturnOp>(op))
        {
            return llvm::all_of(returnOp.getOperandTypes(), [](Type operandType)
            {
                return !llvm::isa<NoneType>(operandType);
            });
        }
        else
        {
            return llvm::all_of(op->getOperandTypes(), [](Type operandType)
            {
                bool result = !llvm::isa<NoneType>(operandType);
                return result;
            });
        }
    }

    bool returnsUnknownType(Operation* op)
    {
        if (op->hasTrait<FunctionOpInterface::Trait>())
        {
            auto function = mlir::dyn_cast_or_null<FunctionOpInterface>(op);
            return llvm::all_of(function.getResultTypes(), [](Type operandType)
            {
                return llvm::isa<NoneType>(operandType);
            });
        }
        else if (auto returnOp = mlir::dyn_cast_or_null<ReturnOp>(op))
        {
            return llvm::all_of(returnOp.getOperandTypes(), [](Type operandType)
            {
                return llvm::isa<NoneType>(operandType);
            });
        }
        else
        {
            return llvm::any_of(op->getResultTypes(), [](Type resultType)
            {
                return llvm::isa<NoneType>(resultType);
            });
        }
    }
}
