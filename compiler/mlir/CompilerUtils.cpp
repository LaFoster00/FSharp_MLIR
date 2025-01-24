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

    mlir::func::FuncOp findFunctionInScope(mlir::Operation* startOp, mlir::StringRef funcName)
    {
        mlir::Operation* currentOp = startOp;

        // Traverse up through parent operations (or regions) to find the closure
        while (currentOp)
        {
            // Check if the current operation has a SymbolTable
            if (currentOp->hasTrait<mlir::OpTrait::SymbolTable>())
            {
                // Try to lookup the closure in the current SymbolTable
                mlir::Operation* function = mlir::SymbolTable::lookupSymbolIn(currentOp, funcName);
                if (auto func_op = mlir::dyn_cast<mlir::func::FuncOp>(function))
                {
                    return func_op; // Found the closure
                }
            }

            // Move to the parent operation
            currentOp = currentOp->getParentOp();
        }

        // If no closure was found, return nullptr
        return nullptr;
    }

    TypeRange getOperands(mlir::Operation* op)
    {
        if (auto function = mlir::dyn_cast_or_null<mlir::FunctionOpInterface>(op))
        {
            return function.getArgumentTypes();
        }
        if (auto returnOp = mlir::dyn_cast_or_null<ReturnOp>(op))
        {
            if (returnOp.getOperand())
                return returnOp.getOperandTypes();
            else
                return {};
        }
        else
        {
            return op->getOperandTypes();
        }
    }

    TypeRange getReturnTypes(mlir::Operation* op)
    {
        if (auto function = mlir::dyn_cast_or_null<mlir::FunctionOpInterface>(op))
        {
            return function.getResultTypes();
        }
        if (auto returnOp = mlir::dyn_cast_or_null<ReturnOp>(op))
        {
            return returnOp->getOperandTypes();
        }
        else
        {
            return op->getResultTypes();
        }
    }

    bool isImplicitTypeInferred(Operation *op)
    {
        return mlir::dyn_cast<ImplicitTypeInference>(op) != nullptr;
    }

    bool someOperandsInferred(Operation* op)
    {
        return llvm::any_of(getOperands(op), [](Type operandType)
        {
            return !llvm::isa<NoneType>(operandType);
        });
    }

    bool allOperandsInferred(Operation* op)
    {
        return llvm::all_of(getOperands(op), [](Type operandType)
        {
            bool result = !llvm::isa<NoneType>(operandType);
            return result;
        });
    }

    bool noOperandsInferred(Operation* op)
    {
        return llvm::all_of(getOperands(op), [](Type operandType)
        {
            return llvm::isa<NoneType>(operandType);
        });
    }

    bool returnsUnknownType(Operation* op)
    {
        return llvm::any_of(getReturnTypes(op), [](Type resultType)
        {
            bool result = llvm::isa<NoneType>(resultType);
            return result;
        });
    }

    bool returnsKnownType(Operation* op)
    {
        return !returnsUnknownType(op);
    }
}
