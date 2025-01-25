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

    bool isImplicitTypeInferred(Operation* op)
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

    void addOrUpdateAttrDictEntry(mlir::Operation* op, mlir::StringRef key, mlir::Attribute value)
    {
        // Get the existing attributes from the operation
        mlir::DictionaryAttr existingAttrs = op->getAttrDictionary();

        // Create a builder to construct attributes
        mlir::MLIRContext* context = op->getContext();
        mlir::OpBuilder builder(context);

        // Prepare a vector to hold updated attributes
        llvm::SmallVector<mlir::NamedAttribute, 8> updatedAttrs;

        bool added_inplace = false;
        // Copy existing attributes to the updated list, except the one being updated
        for (const auto& attr : existingAttrs)
        {
            if (attr.getName() != key)
            {
                updatedAttrs.push_back(attr);
            }
            else
            {
                // Add or update the attribute with the provided key and value
                updatedAttrs.push_back(builder.getNamedAttr(key, value));
                added_inplace = true;
            }
        }

        if (!added_inplace)
            updatedAttrs.push_back(builder.getNamedAttr(key, value));

        // Set the updated attribute dictionary back on the operation
        op->setAttrs(mlir::DictionaryAttr::get(context, updatedAttrs));
    }

    void deleteAttrDictEntry(mlir::Operation* op, mlir::StringRef key)
    {
        // Get the existing attributes from the operation
        mlir::DictionaryAttr existingAttrs = op->getAttrDictionary();

        // If the attribute dictionary is empty, there's nothing to delete
        if (!existingAttrs || existingAttrs.empty())
            return;

        // Create a vector to hold the updated attributes
        llvm::SmallVector<mlir::NamedAttribute, 8> updatedAttrs;

        // Copy all attributes except the one with the specified key
        for (const auto& attr : existingAttrs)
        {
            if (attr.getName() != key)
            {
                updatedAttrs.push_back(attr);
            }
        }

        // Create a new dictionary attribute from the updated list
        mlir::MLIRContext* context = op->getContext();
        op->setAttrs(mlir::DictionaryAttr::get(context, updatedAttrs));
    }

    // Helper function to create a global memref for the string.
    Value createGlobalMemrefForString(Location loc, StringRef stringValue,
                                             OpBuilder& builder, ModuleOp module, Operation* op)
    {
        // Check if the global already exists in the module.
        std::string globalName = "__printf_format_" + std::to_string(hash_value(stringValue));
        auto existingGlobal = module.lookupSymbol<memref::GlobalOp>(globalName);
        if (existingGlobal)
            return builder.create<memref::GetGlobalOp>(
                loc, existingGlobal.getType(), existingGlobal.getName());

        // Create a global memref.
        builder.setInsertionPointToStart(module.getBody());
        // Prepare data attribute for the global.
        std::vector<char8_t> data{stringValue.begin(), stringValue.end()};
        data.push_back('\0');
        auto type = mlir::RankedTensorType::get({static_cast<int64_t>(data.size())}, builder.getI8Type());
        auto dataAttribute = mlir::DenseElementsAttr::get(type, llvm::ArrayRef(data));
        // Create the global using the data attribute.
        auto globalOp = builder.create<memref::GlobalOp>(
            loc,
            globalName,
            builder.getStringAttr("private"),
            MemRefType::get(type.getShape(), type.getElementType()),
            dataAttribute,
            /*constant=*/true, /*alignment=*/nullptr);

        // Return a GetGlobalOp for this global.
        builder.setInsertionPoint(op);
        return builder.create<memref::GetGlobalOp>(
            loc, globalOp.getType(), globalOp.getName());
    }
}
