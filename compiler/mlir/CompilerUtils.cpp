//
// Created by lasse on 1/23/25.
//
#include "compiler/CompilerUtils.h"

#include <ast/ASTNode.h>
#include <utils/Utils.h>

namespace mlir::fsharp::utils
{
    std::string getTypeString(mlir::Type type)
    {
        std::string typeStr;
        llvm::raw_string_ostream rso(typeStr);
        type.print(rso);
        return rso.str();
    }

    mlir::Location loc(const fsharpgrammar::ast::Range& range,
                       const std::string_view filename,
                       mlir::MLIRContext* context)
    {
        return mlir::FileLineColLoc::get(
            StringAttr::get(context, filename),
            range.start_line(),
            range.start_column());
    }

    mlir::Location loc(const fsharpgrammar::ast::IASTNode& node,
                       const std::string_view filename,
                       mlir::MLIRContext* context)
    {
        return loc(node.get_range(), filename, context);
    }

    mlir::Location loc(const fsharpgrammar::ast::INodeAlternative& node_alternative,
                       const std::string_view filename,
                       mlir::MLIRContext* context)
    {
        return loc(node_alternative.get_range(), filename, context);
    }

    mlir::Type getMLIRType(const std::string_view type_name, mlir::MLIRContext* context)
    {
        if (type_name == "int")
            return IntegerType::get(context, 32, IntegerType::Signed);
        if (type_name == "float")
            return FloatType::getF64(context);
        if (type_name == "bool")
            return IntegerType::get(context, 8, IntegerType::Signless);
        if (type_name == "string")
            return mlir::UnrankedTensorType::get(IntegerType::get(context, 8, IntegerType::Signless));
        assert(false && "Type not supported!");
    }

    mlir::Type getMLIRType(const fsharpgrammar::ast::Type& type, mlir::MLIRContext* context, mlir::Location loc)
    {
        return std::visit<mlir::Type>(
            ::utils::overloaded{
                [&](const fsharpgrammar::ast::Type::Fun& t)
                {
                    mlir::emitError(loc, "Parameters with function types not supported!");
                    return NoneType::get(context);
                },
                [&](const fsharpgrammar::ast::Type::Tuple& t)
                {
                    mlir::emitError(loc, "Tuples not supported!");
                    return NoneType::get(context);
                },
                [&](const fsharpgrammar::ast::Type::Postfix& t)
                {
                    mlir::emitError(loc, "Postfix types not supported!");
                    return NoneType::get(context);
                },
                [&](const fsharpgrammar::ast::Type::Array& t)
                {
                    mlir::emitError(loc, "Arrays not supported!");
                    return NoneType::get(context);
                },
                [&](const fsharpgrammar::ast::Type::Paren& t)
                {
                    return getMLIRType(*t.type, context, loc);
                },
                [&](const fsharpgrammar::ast::Type::Var& t)
                {
                    return getMLIRType(t.ident->ident, context);
                },
                [&](const fsharpgrammar::ast::Type::LongIdent& t)
                {
                    mlir::emitError(loc, "Namespace- and module-types not supported!");
                    return NoneType::get(context);
                },
                [&](const fsharpgrammar::ast::Type::Anon& t)
                {
                    mlir::emitError(loc, "Anonymous types not supported!");
                    return NoneType::get(context);
                },
                [&](const fsharpgrammar::ast::Type::StaticConstant& t)
                {
                    mlir::emitError(loc, "Static constants not supported!");
                    return NoneType::get(context);
                },
                [&](const fsharpgrammar::ast::Type::StaticNull& t)
                {
                    mlir::emitError(loc, "Static null not supported!");
                    return NoneType::get(context);
                }
            }, type.type);
    }

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

    llvm::SmallVector<std::tuple<char, mlir::Type>, 4> getFormatSpecifiedTypes(
        llvm::StringRef format, mlir::MLIRContext* context)
    {
        llvm::SmallVector<std::tuple<char, mlir::Type>, 4> types;

        // Define a simple state machine to parse the format string
        for (size_t i = 0; i < format.size(); ++i)
        {
            if (format[i] == '%')
            {
                ++i; // Advance to the character after '%'

                // Skip flags and width/precision modifiers (e.g., "%-10.3d").
                while (i < format.size() && (format[i] == '-' || format[i] == '+' ||
                    format[i] == ' ' || format[i] == '#' ||
                    format[i] == '0' || std::isdigit(format[i]) ||
                    format[i] == '.'))
                {
                    ++i;
                }

                // Ensure we haven't reached the end of the string.
                if (i >= format.size())
                {
                    break;
                }

                // Check the specifier and map to an MLIR type.
                switch (format[i])
                {
                case 'b':
                    types.push_back({format[i], IntegerType::get(context, 1)});
                    break;
                case 's': // String
                    types.push_back({format[i], mlir::UnrankedTensorType::get(IntegerType::get(context, 8))});
                    break;
                case 'c': // Character
                    types.push_back({format[i], IntegerType::get(context, 8, IntegerType::Signless)});
                    break;
                case 'i': // Integer
                case 'd':
                case 'u':
                case 'o':
                case 'x':
                case 'X':
                    types.push_back({
                        format[i], IntegerType::get(context, 64, IntegerType::SignednessSemantics::Signless)
                    });
                    break;
                case 'f':
                case 'F':
                case 'e':
                case 'E':
                case 'g':
                case 'G': // Float
                    types.push_back({format[i], mlir::FloatType::getF64(context)}); // Default to 32-bit float
                    break;
                case '%': // Literal '%' (no type, skip)
                    break;
                case 'A':
                // Formatted using structured plain text formatting with the default layout settings. Not actually supported.
                // TODO implement and provide a default method for printing arrays and lists
                case 'a':
                // Requires two arguments: a formatting function accepting a context parameter and the value, and the particular value to print
                case 't':
                //Requires one argument: a formatting function accepting a context parameter that either outputs or returns the appropriate text
                case 'O': // Box object and call System.Object.ToString()
                default:
                    llvm::errs() << "Unsupported format specifier: " << format[i] << "\n";
                    break;
                }
            }
        }

        return types;
    }
}
