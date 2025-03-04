//
// Created by lasse on 1/7/25.
//
#include "compiler/ASTToMLIR.h"

#include <ranges>

#include "compiler/FSharpDialect.h"
#include <span>
#include <fmt/color.h>

#include <ast/ASTNode.h>
#include <ast/Range.h>
#include <compiler/CompilerUtils.h>

#include "Grammar.h"


#include "boost/algorithm/string.hpp"

using namespace fsharpgrammar;
using namespace mlir::fsharp;

namespace fsharp::compiler
{
    mlir::Value convertTensorToDType(mlir::OpBuilder& builder, mlir::Location loc, mlir::Value value, mlir::Type toType)
    {
        if (!mlir::tensor::CastOp::areCastCompatible(value.getType(), toType))
        {
            mlir::emitError(loc, fmt::format("Can't cast tensor from {} to {}!",
                                             mlir::fsharp::utils::getTypeString(value.getType()),
                                             mlir::fsharp::utils::getTypeString(toType)));
            return value;
        }

        return builder.create<mlir::tensor::CastOp>(loc, toType, value);
    }

    mlir::Value genCast(mlir::OpBuilder& builder, mlir::Location loc, mlir::Value value,
                        mlir::Type dstTp)
    {
        const mlir::Type srcTp = value.getType();
        if (srcTp == dstTp)
            return value;

        if (value.getType().isa<mlir::TensorType>())
            return convertTensorToDType(builder, loc, value, dstTp.dyn_cast<mlir::TensorType>());

        // int <=> index
        if (isa<mlir::IndexType>(srcTp) || isa<mlir::IndexType>(dstTp))
            return builder.create<mlir::arith::IndexCastOp>(loc, dstTp, value);

        const auto srcIntTp = dyn_cast_or_null<mlir::IntegerType>(srcTp);
        const bool isUnsignedCast = srcIntTp ? srcIntTp.isUnsigned() : false;
        return mlir::convertScalarToDtype(builder, loc, value, dstTp, isUnsignedCast);
    }

    class MLIRGenImpl
    {
    public:
        explicit MLIRGenImpl(mlir::MLIRContext& context, std::string_view source_filename) : filename(source_filename),
            builder(&context)
        {
        }

        mlir::ModuleOp mlirGen(const ast::Main& main_ast)
        {
            // We create an empty MLIR module and codegen functions one at a time and
            // add them to the module.
            fileModule = mlir::ModuleOp::create(builder.getUnknownLoc(), filename);

            for (auto& f : main_ast.modules_or_namespaces)
                mlirGen(*f);

            // Verify the module after we have finished constructing it, this will check
            // the structural properties of the IR and invoke any specific verifiers we
            // have on the Toy operations.
            if (failed(mlir::verify(fileModule)))
            {
                fileModule.emitError("module verification error");
#ifndef DEBUG
                return nullptr;
#endif
            }

            return fileModule;
        }

    private:
        std::string_view filename;

        /// A "module" matches a fsharp source file: containing a list of functions.
        mlir::ModuleOp fileModule;

        /// An optional cached op that can be used to pass a specific op to functions which might be called in a context
        /// where accessing the actual mlir is not possible (e.g. branch generators of ScfIfOp)
        mlir::fsharp::ClosureOp lastDeclaredFunction;

        /// The builder is a helper class to create IR inside a function. The builder
        /// is stateful, in particular it keeps an "insertion point": this is where
        /// the next operations will be introduced.
        mlir::OpBuilder builder;

        /// The symbol table maps a variable name to a value in the current scope.
        /// Entering a function creates a new scope, and the function arguments are
        /// added to the mapping. When the processing of a function is terminated, the
        /// scope is destroyed and the mappings created in this scope are dropped.
        llvm::ScopedHashTable<llvm::StringRef, mlir::Value> symbolTable;

    private:
        mlir::Location loc(const ast::Range& range)
        {
            return mlir::fsharp::utils::loc(range, filename, builder.getContext());
        }

        mlir::Location loc(const ast::IASTNode& node)
        {
            return loc(node.get_range());
        }

        mlir::Location loc(const ast::INodeAlternative& node_alternative)
        {
            return loc(node_alternative.get_range());
        }

        /// Declare a variable in the current scope, return success if the variable
        /// wasn't declared yet.
        llvm::LogicalResult declare(const llvm::StringRef var, mlir::Value value)
        {
            if (symbolTable.count(var))
            {
                mlir::emitError(value.getLoc(),
                                fmt::format(
                                    "Cannot declare variable '{}' with same name twice in the same scope!",
                                    var.str()));
                return mlir::failure();
            }
            symbolTable.insert(var, value);
            return mlir::success();
        }

        mlir::Type getSmallestCommonType(mlir::ArrayRef<mlir::Value> values)
        {
            mlir::Type smallest_type = nullptr;
            for (auto value : values)
            {
                // If the value is of type None, and no type was found yet, set the smallest type to i32
                if (mlir::isa<mlir::NoneType>(value.getType()) && smallest_type == nullptr)
                {
                    smallest_type = builder.getIntegerType(32, true);
                    continue;
                }
                // Check if we need to upcast the results to a larger type
                if (smallest_type == nullptr ||
                    (smallest_type.dyn_cast<mlir::IntegerType>()
                        && smallest_type.getIntOrFloatBitWidth() < value.getType().getIntOrFloatBitWidth()) ||
                    (smallest_type.dyn_cast<mlir::IntegerType>()
                        && smallest_type.dyn_cast<mlir::FloatType>())
                    ||
                    (smallest_type.dyn_cast<mlir::FloatType>()
                        && smallest_type.getIntOrFloatBitWidth() < value.getType().getIntOrFloatBitWidth()))
                {
                    smallest_type = value.getType();
                }
            }
            return smallest_type;
        }

    private:
        mlir::fsharp::ClosureOp createEntryPoint()
        {
            // Create a scope in the symbol table to hold variable declarations.
            llvm::ScopedHashTableScope<llvm::StringRef, mlir::Value> varScope(symbolTable);

            builder.setInsertionPointToEnd(fileModule.getBody());

            // Define the function type (no arguments, no return values)
            mlir::FunctionType funcType = builder.getFunctionType({}, {});

            // Create the function
            auto func = builder.create<mlir::fsharp::ClosureOp>(fileModule.getLoc(), "main", funcType);

            // Add a basic block to the function
            mlir::Block& entryBlock = func.front();

            // Set the insertion point to the start of the basic block
            builder.setInsertionPointToStart(&entryBlock);

            return func;
        }

        mlir::ModuleOp mlirGen(const ast::ModuleOrNamespace& module_or_namespace)
        {
            // Create a scope in the symbol table to hold variable declarations.
            llvm::ScopedHashTableScope<llvm::StringRef, mlir::Value> varScope(symbolTable);

            mlir::ModuleOp m;
            switch (module_or_namespace.type)
            {
            case fsharpgrammar::ast::ModuleOrNamespace::Type::Namespace:
                throw std::runtime_error("Namespaces not supported!");
            case fsharpgrammar::ast::ModuleOrNamespace::Type::NamedModule:
                m = builder.create<mlir::ModuleOp>(loc(module_or_namespace.range),
                                                   module_or_namespace.name.value()->get_as_string());
                break;
            case fsharpgrammar::ast::ModuleOrNamespace::Type::AnonymousModule:
                m = fileModule;
                break;
            }
            builder.setInsertionPointToEnd(m.getBody());

            auto func = createEntryPoint();

            for (auto& module_decl : module_or_namespace.moduleDecls)
            {
                std::visit([&](auto& obj) { mlirGen(obj); },
                           module_decl->declaration);
            }


            builder.setInsertionPointToEnd(&func.getBlocks().back());
            builder.create<mlir::fsharp::ReturnOp>(builder.getUnknownLoc());

            builder.setInsertionPointToEnd(fileModule.getBody());

            return m;
        }

        void mlirGen(const ast::ModuleDeclaration::Open& open)
        {
            throw std::runtime_error("Open statement not supported!");
        }

        mlir::ModuleOp mlirGen(const ast::ModuleDeclaration::NestedModule& nested_module)
        {
            throw std::runtime_error("Nested modules not supported!");
        }

        std::optional<mlir::Value> mlirGen(const ast::ModuleDeclaration::Expression& expression)
        {
            return mlirGen(*expression.expression);
        }

        std::optional<mlir::Value> mlirGen(const std::vector<ast::ast_ptr<ast::Expression>>& expressions)
        {
            std::optional<mlir::Value> value{};
            for (auto& expression : expressions)
            {
                value = mlirGen(*expression);
                if (!value.has_value())
                    continue;

                if (value.value() == nullptr)
                    return value;
            }
            return value;
        }

        std::optional<mlir::Value> mlirGen(const ast::Expression& expression)
        {
            if (auto append = std::get_if<ast::Expression::Append>(&expression.expression))
            {
                // Get the generation result of the append expression
                const auto result = mlirGen(*append);
                // Return the value if a value was generated
                if (const auto value = std::get_if<std::optional<mlir::Value>>(&result))
                {
                    return *value;
                }
                // Return invalid state in case the generation failed else nullopt
                if (const auto logical_result = std::get_if<llvm::LogicalResult>(&result))
                {
                    if (llvm::succeeded(*logical_result))
                        return std::optional<mlir::Value>{};
                    return mlir::Value(nullptr);
                }
            }
            if (auto constant = std::get_if<ast::Expression::Constant>(&expression.expression))
                return mlirGen(*constant);
            if (auto let = std::get_if<ast::Expression::Let>(&expression.expression))
            {
                auto result = mlirGen(*let);
                // If the function definition was not successful return the invalid state nullptr
                if (auto logical_result = std::get_if<llvm::LogicalResult>(&result))
                    return logical_result->succeeded()
                               ? std::optional<mlir::Value>{}
                               : mlir::Value(nullptr);
                else
                    return std::get<mlir::Value>(result);
            }
            if (auto ident = std::get_if<ast::Expression::Ident>(&expression.expression))
                return mlirGen(*ident);
            if (auto op = std::get_if<ast::Expression::OP>(&expression.expression))
                return mlirGen(*op);
            if (auto if_then_else = std::get_if<ast::Expression::IfThenElse>(&expression.expression))
                return mlirGen(*if_then_else);
            // Paren expressions can be skipped since we only care about the inner expression
            if (auto paren = std::get_if<ast::Expression::Paren>(&expression.expression))
                return mlirGen(*paren->expression);
            if (auto unary = std::get_if<ast::Expression::Unary>(&expression.expression))
                return mlirGen(*unary);

            mlir::emitError(loc(expression.get_range()), "Expression not supported!");
        }

        std::span<const ast::ast_ptr<ast::Expression>> getFunctionArgs(const ast::Expression::Append& append)
        {
            return {std::next(append.expressions.begin()), append.expressions.end()};
        }

        mlir::Value getFunctionArgValue(const ast::Expression::Append& append, size_t index)
        {
            auto args = getFunctionArgs(append);
            if (args.size() <= index)
            {
                mlir::emitError(loc(append.get_range()), "Function argument index out of bounds!");
                return nullptr;
            }
            auto value = mlirGen(*args[index]);
            if (!value.has_value())
            {
                mlir::emitError(loc(append.get_range()), "Function argument value does not return a value!");
                return nullptr;
            }
            return value.value();
        }

        std::optional<llvm::SmallVector<mlir::Value, 4>> getFunctionArgValues(
            const ast::Expression::Append& append, bool skip_first)
        {
            // Codegen the operands first
            llvm::SmallVector<mlir::Value, 4> operands;
            for (auto [i, expr] : llvm::enumerate(getFunctionArgs(append)))
            {
                if (i == 0 && skip_first)
                    continue;

                const auto arg = mlirGen(*expr);
                if (!arg.has_value())
                {
                    fmt::print(fmt::fg(fmt::color::orange_red), "Function argument value does not return a value! {}",
                               ::utils::to_string(expr->get_range()));
                    return {};
                };
                operands.push_back(arg.value());
            }
            return operands;
        }

        std::variant<std::optional<mlir::Value>, llvm::LogicalResult> mlirGen(const ast::Expression::Append& append)
        {
            if (append.isFunctionCall)
            {
                std::string func_name;
                if (auto ident = std::get_if<ast::Expression::Ident>(&append.expressions.front()->expression))
                {
                    func_name = ident->ident->ident;
                }
                else if (auto long_ident = std::get_if<ast::Expression::LongIdent>(
                    &append.expressions.front()->expression))
                {
                    func_name = long_ident->longIdent->get_as_string();
                }

                if (func_name == "print" || func_name == "printf" || func_name == "printfn")
                    return generatePrint(append);
                if (func_name == "assert")
                    return generateAssert(append);

                return declareFunctionCall(append, func_name);
            }
            else
            {
                mlir::emitError(loc(append.get_range()), "Append not supported for anything but function invocations!");
                return mlir::Value(nullptr);
            }
        }

        /// Find a closure with the given name in the current scope or parent scopes.
        mlir::fsharp::ClosureOp findClosureInScope(mlir::Operation* startOp, mlir::StringRef closureName)
        {
            mlir::Operation* currentOp = startOp;
            if (currentOp == nullptr)
                currentOp = lastDeclaredFunction;

            //mlir::emitError(currentOp->getLoc(), "Looking into the block") << *currentOp;

            // Traverse up through parent operations (or regions) to find the closure
            while (currentOp)
            {
                // Check if the current operation has a SymbolTable
                if (currentOp->hasTrait<mlir::OpTrait::SymbolTable>())
                {
                    // Try to lookup the closure in the current SymbolTable
                    mlir::Operation* closure = mlir::SymbolTable::lookupSymbolIn(currentOp, closureName);
                    if (closure)
                    {
                        if (auto closure_op = mlir::dyn_cast_or_null<mlir::fsharp::ClosureOp>(closure))
                        {
                            return closure_op; // Found the closure
                        }
                    }
                }

                // Move to the parent operation
                currentOp = currentOp->getParentOp();
            }

            // If no closure was found, return nullptr
            return nullptr;
        }

        std::optional<mlir::Value> declareFunctionCall(const ast::Expression::Append& append,
                                                       const std::string& func_name)
        {
            auto location = loc(append.get_range());

            auto args = getFunctionArgValues(append, false);
            if (!args.has_value())
                return nullptr;
            mlir::ValueRange arg_values = args.value();
            auto closureOp = findClosureInScope(builder.getBlock()->getParentOp(), mlir::StringRef(func_name));
            if (closureOp)
                if (closureOp.getNumResults() == 0)
                {
                    builder.create<mlir::fsharp::CallOp>(location, closureOp, arg_values);
                    return {};
                }
                else
                    return builder.create<mlir::fsharp::CallOp>(location, closureOp, arg_values)->getResult(0);
            else
            {
                mlir::emitError(loc(append),
                                "Could not find function with name '" + func_name + "' in the current scope!");
                return nullptr;
            }
        }

        llvm::LogicalResult generatePrint(const ast::Expression::Append& append)
        {
            auto args = getFunctionArgValues(append, true);
            if (!args.has_value())
                return llvm::failure();
            if (auto fmt_string_constant = std::get_if<ast::Expression::Constant>(&append.expressions[1]->expression))
            {
                if (auto fmt_string = std::get_if<std::string>(&fmt_string_constant->constant->value.value()))
                {
                    auto fmt_string_attr = builder.getStringAttr(*fmt_string);
                    builder.create<mlir::fsharp::PrintOp>(loc(append.get_range()), fmt_string_attr,
                                                          mlir::ValueRange(args.value()));
                    return llvm::success();
                }
            }
            auto op = builder.create<mlir::fsharp::PrintOp>(loc(append.get_range()), builder.getStringAttr(""),
                                                            mlir::ValueRange{});
            mlir::fsharp::utils::addOrUpdateAttrDictEntry(op.getOperation(), "malformed", builder.getUnitAttr());
            mlir::emitError(loc(append.get_range()), "First argument of print must be a formattable string literal!");
            return llvm::failure();
        }

        llvm::LogicalResult generateAssert(const ast::Expression::Append& append)
        {
            auto num_args = append.expressions.size();
            // Assert must have at max two arguments
            // The first argument is the function name so we need to allways add 1
            if (num_args != 2)
            {
                if (num_args == 3)
                {
                    // Check if we are holding a string literal as the second function argument
                    if (auto constant = std::get_if<ast::Expression::Constant>(&append.expressions[2]->expression))
                    {
                        if (!std::holds_alternative<std::string>(constant->constant->value.value()))
                        {
                            emitError(loc(append), "Second argument of assert must be a string literal!");
                            return llvm::failure();
                        }
                    }
                    else
                    {
                        emitError(loc(append), "Second argument of assert must be a string literal!");
                        return llvm::failure();
                    }
                }
                else
                {
                    emitError(loc(append.get_range()), "Assert must have at max two arguments!");
                    return llvm::failure();
                }
            }
            // Get the condition value
            auto condition = mlirGen(*append.expressions[1]);
            if (!condition.has_value() || condition.value() == nullptr)
            {
                emitError(loc(append), "Invalid condition value!");
                return llvm::failure();
            }
            // Only build assert with the explicit message if we have three arguments
            if (num_args == 3)
            {
                // Get the message string attr
                auto message = builder.getStringAttr(
                    std::get<std::string>(
                        std::get<ast::Expression::Constant>(
                            append.expressions[2]->expression)
                        .constant->value.value()));
                builder.create<mlir::fsharp::AssertOp>(loc(append.get_range()), condition.value(), message);
            }
            else
            {
                builder.create<mlir::fsharp::AssertOp>(loc(append.get_range()), condition.value());
            }
            return llvm::success();
        }

        mlir::Value mlirGen(const ast::Expression::OP& op)
        {
            switch (op.type)
            {
            case ast::Expression::OP::Type::LOGICAL:
                return getLogicalOp(op);
                break;
            // return getLogicalType(std::get<0>(op));
            case ast::Expression::OP::Type::EQUALITY:
                return getEqualityOp(op);
                break;
            // return getEqualityType(std::get<1>(operators));
            case ast::Expression::OP::Type::RELATION:
                return getRelationOp(op);
                break;
            // return getRelationType(std::get<2>(operators));
            case ast::Expression::OP::Type::ARITHMETIC:
                return getArithmeticOp(op);
            default:
                mlir::emitError(loc(op.get_range()), "No initializer given to op.type!");
                break;
            }
            return nullptr;
        }

        static bool isAffineContext(mlir::Block* block)
        {
            // Traverse up the parent operations to check for an affine context.
            for (mlir::Operation* op = block->getParentOp(); op != nullptr; op = op->getParentOp())
            {
                // Check if the operation is an affine construct.
                if (llvm::isa<mlir::affine::AffineForOp>(op) || llvm::isa<mlir::affine::AffineIfOp>(op))
                {
                    return true; // Found an affine context.
                }
            }
            return false; // No affine context found.
        }

        // Returns true if the branch returns a value
        static bool ProcessThenBranch(mlir::OpBuilder& builder, const ast::Expression::IfThenElse& if_then_else,
                                      bool is_affine_context, MLIRGenImpl& mlirGen)
        {
            mlir::SmallVector<std::optional<mlir::Value>, 4> then_results;
            for (auto& then_expr : if_then_else.then)
            {
                then_results.emplace_back(mlirGen.mlirGen(*then_expr));
            }

            bool returning_value = then_results.back().has_value();
            if (returning_value || !is_affine_context)
            {
                if (is_affine_context)
                    builder.create<mlir::affine::AffineYieldOp>(then_results.back()->getLoc(),
                                                                mlir::ValueRange{then_results.back().value()});
                else
                    builder.create<mlir::scf::YieldOp>(
                        returning_value
                            ? then_results.back()->getLoc()
                            : mlirGen.loc(if_then_else),
                        returning_value
                            ? mlir::ValueRange{then_results.back().value()}
                            : mlir::ValueRange{});
            }
            return returning_value;
        }

        // Returns false on failure
        static llvm::LogicalResult ProcessElseBranch(mlir::OpBuilder& builder,
                                                     const ast::Expression::IfThenElse& if_then_else,
                                                     bool returning_value,
                                                     bool is_affine_context,
                                                     MLIRGenImpl& mlirGen)
        {
            mlir::SmallVector<std::optional<mlir::Value>, 4> else_results;
            for (auto& else_expr : if_then_else.else_expr.value())
            {
                else_results.emplace_back(mlirGen.mlirGen(*else_expr));
            }
            // If the last statement in the else block returns a value we need to yield it in case it is used as a return value
            // e.g let a = if true then 1 else 2
            // Check if this branch is actually returning a value. If not we need to throw an error
            if (returning_value && (else_results.empty() || !else_results.back().has_value()))
            {
                mlir::emitError(mlirGen.loc(if_then_else),
                                "Either both branches or none of the branches must return a value!");
                return llvm::failure();
            }

            if (returning_value || !is_affine_context)
                if (is_affine_context)
                    builder.create<mlir::affine::AffineYieldOp>(else_results.back()->getLoc(),
                                                                mlir::ValueRange{else_results.back().value()});
                else
                    builder.create<mlir::scf::YieldOp>(returning_value
                                                           ? else_results.back()->getLoc()
                                                           : mlirGen.loc(if_then_else),
                                                       returning_value
                                                           ? mlir::ValueRange{else_results.back().value()}
                                                           : mlir::ValueRange{});
            return llvm::success();
        }

        std::optional<mlir::Value> mlirGen(const ast::Expression::IfThenElse& if_then_else)
        {
            // Create a scope in the symbol table to hold variable declarations.
            llvm::ScopedHashTableScope<llvm::StringRef, mlir::Value> varScope(symbolTable);

            auto condition = mlirGen(*if_then_else.condition);
            if (!condition.has_value())
            {
                mlir::emitError(loc(if_then_else), "Condition did not return value!");
                return nullptr;
            }
            if (condition->getType() != builder.getI1Type())
            {
                mlir::emitError(loc(if_then_else), "Condition does not evaluate to a boolean value!");
                return nullptr;
            }

            const bool is_affine_context = isAffineContext(builder.getBlock());

            mlir::OpBuilder::InsertionGuard guard(builder);
            if (is_affine_context)
            {
                auto affine_if = builder.create<mlir::affine::AffineIfOp>(loc(if_then_else),
                                                                          mlir::IntegerSet::getEmptySet(
                                                                              0, 0, builder.getContext()),
                                                                          // TODO implement the correct logic for this
                                                                          mlir::ValueRange{condition.value()},
                                                                          if_then_else.else_expr.has_value());


                builder.setInsertionPointToStart(affine_if.getThenBlock());
                const bool returning_value = ProcessThenBranch(builder, if_then_else, true, *this);

                if (if_then_else.else_expr.has_value())
                {
                    builder.setInsertionPointToStart(affine_if.getElseBlock());
                    if (llvm::failed(ProcessElseBranch(builder, if_then_else, returning_value, true, *this)))
                        return nullptr;
                }

                if (returning_value)
                {
                    return affine_if.getResult(0);
                }
                return {};
            }
            else
            {
                bool returning_value;
                llvm::LogicalResult else_result = llvm::success();
                auto scf_if = builder.create<mlir::scf::IfOp>(
                    loc(if_then_else),
                    condition.value(),
                    // Then builder with inferred return type
                    [&](mlir::OpBuilder& builder, mlir::Location loc)
                    {
                        returning_value = ProcessThenBranch(builder, if_then_else, false, *this);
                    },
                    [&](mlir::OpBuilder& builder, mlir::Location loc)
                    {
                        else_result = ProcessElseBranch(builder, if_then_else, returning_value, false, *this);
                    }
                );
                // Check if the construction worked as expected
                if (llvm::failed(else_result))
                    return nullptr;

                if (returning_value)
                    return scf_if.getResult(0);
                return {};
            }
        }

        mlir::Value mlirGen(const ast::Expression::Unary& unary)
        {
            auto expression = mlirGen(*unary.expression);
            if (!expression.has_value() || expression.value() == nullptr)
            {
                mlir::emitError(loc(unary), "Unary expression did not return value!");
                return nullptr;
            }

            mlir::fsharp::ConstantOp factor = nullptr;
            mlir::Value result = expression.value();
            switch (unary.type)
            {
            case ast::Expression::Unary::Type::PLUS:
                break;
            case ast::Expression::Unary::Type::MINUS:
                result = builder.create<mlir::fsharp::NegateOp>(loc(unary), expression.value());
                break;
            case ast::Expression::Unary::Type::NOT:
                result = builder.create<mlir::fsharp::NotOp>(loc(unary), expression.value());
                break;
            }
            return result;
        }

        mlir::Value getRelationOp(const ast::Expression::OP& op)
        {
            auto relationOps = std::get<std::vector<ast::Expression::OP::RelationType>>(op.ops);
            mlir::SmallVector<mlir::Value, 4> operands;
            for (auto& expression : op.expressions)
            {
                auto result = mlirGen(*expression);
                if (!result.has_value())
                {
                    mlir::emitError(loc(op.get_range()), "Operand did not return value!");
                    return nullptr;
                }
                operands.push_back(result.value());
            }

            mlir::Value result = operands[0];
            for (auto [index, relation_op] : llvm::enumerate(relationOps))
            {
                switch (relation_op)
                {
                case ast::Expression::OP::RelationType::LESS:
                    result = builder.create<mlir::fsharp::LessOp>(loc(op), result, operands[index + 1]);
                    break;
                case ast::Expression::OP::RelationType::LESS_EQUAL:
                    result = builder.create<mlir::fsharp::LessEqualOp>(loc(op), result, operands[index + 1]);
                    break;
                case ast::Expression::OP::RelationType::GREATER:
                    result = builder.create<mlir::fsharp::GreaterOp>(loc(op), result, operands[index + 1]);
                    break;
                case ast::Expression::OP::RelationType::GREATER_EQUAL:
                    result = builder.create<mlir::fsharp::GreaterEqualOp>(loc(op), result, operands[index + 1]);
                    break;
                }
            }

            return result;
        }


        mlir::Value getEqualityOp(const ast::Expression::OP& op)
        {
            auto equalityOps = std::get<std::vector<ast::Expression::OP::EqualityType>>(op.ops);

            mlir::SmallVector<mlir::Value, 4> operands;
            for (auto& expression : op.expressions)
            {
                auto result = mlirGen(*expression);
                if (!result.has_value())
                {
                    mlir::emitError(loc(op.get_range()), "Operand did not return value!");
                    return nullptr;
                }
                operands.push_back(result.value());
            }

            mlir::Value result = operands[0];
            for (auto [index, equality_op] : llvm::enumerate(equalityOps))
            {
                switch (equality_op)
                {
                case ast::Expression::OP::EqualityType::EQUAL:
                    result = builder.create<mlir::fsharp::EqualOp>(loc(op), result, operands[index + 1]);
                    break;
                case ast::Expression::OP::EqualityType::NOT_EQUAL:
                    result = builder.create<mlir::fsharp::NotEqualOp>(loc(op), result, operands[index + 1]);
                    break;
                default: mlir::emitError(loc(op.get_range()), "No equality operator found in OP!");
                }
            }

            return result;
        }

        mlir::Value getLogicalOp(const ast::Expression::OP& op)
        {
            auto logicalOps = std::get<std::vector<ast::Expression::OP::LogicalType>>(op.ops);

            mlir::SmallVector<mlir::Value, 4> operands;
            for (auto& expression : op.expressions)
            {
                auto result = mlirGen(*expression);
                if (!result.has_value())
                {
                    mlir::emitError(loc(op.get_range()), "Operand did not return value!");
                    return nullptr;
                }
                operands.push_back(result.value());
            }

            mlir::Value result = operands[0];
            for (auto [index, logical_op] : llvm::enumerate(logicalOps))
            {
                switch (logical_op)
                {
                case ast::Expression::OP::LogicalType::AND:
                    result = builder.create<mlir::fsharp::AndOp>(loc(op), result, operands[index + 1]);
                    break;
                case ast::Expression::OP::LogicalType::OR:
                    result = builder.create<mlir::fsharp::OrOp>(loc(op), result, operands[index + 1]);
                    break;
                default: mlir::emitError(loc(op.get_range()), "No logical operator found in OP!");
                }
            }

            return result;
        }

        mlir::Value getArithmeticOp(const ast::Expression::OP& op)
        {
            if (auto arithmeticOps = std::get_if<std::vector<ast::Expression::OP::ArithmeticType>>(&op.ops))
            {
                mlir::SmallVector<mlir::Value, 4> operands;
                for (auto& expression : op.expressions)
                {
                    auto result = mlirGen(*expression);
                    if (!result.has_value())
                    {
                        mlir::emitError(loc(op.get_range()), "Operand did not return value!");
                        return nullptr;
                    }
                    operands.push_back(result.value());
                }

                mlir::Value result = operands[0];
                for (auto [index, arith_op] : llvm::enumerate(*arithmeticOps))
                {
                    switch (arith_op)
                    {
                    case ast::Expression::OP::ArithmeticType::ADD:
                        result = builder.create<mlir::fsharp::AddOp>(loc(op), result, operands[index + 1]);
                        break;
                    case ast::Expression::OP::ArithmeticType::SUBTRACT:
                        result = builder.create<mlir::fsharp::SubOp>(loc(op), result, operands[index + 1]);
                        break;
                    case ast::Expression::OP::ArithmeticType::MULTIPLY:
                        result = builder.create<mlir::fsharp::MulOp>(loc(op), result, operands[index + 1]);
                        break;
                    case ast::Expression::OP::ArithmeticType::DIVIDE:
                        result = builder.create<mlir::fsharp::DivOp>(loc(op), result, operands[index + 1]);
                        break;
                    case ast::Expression::OP::ArithmeticType::MODULO:
                        result = builder.create<mlir::fsharp::ModOp>(loc(op), result, operands[index + 1]);
                        break;
                    }
                }

                return result;
            }

            mlir::emitError(loc(op.get_range()), "No arithmetic operator found in OP!");
            return nullptr;
        }

        mlir::Value mlirGen(const ast::Expression::Constant& constant)
        {
            return mlirGen(*constant.constant);
        }

        std::tuple<std::reference_wrapper<const std::string>, std::reference_wrapper<const std::string>> getLetArg(
            const ast::Pattern::PatternType& pattern)
        {
            static const std::string no_type = "";
            if (auto named = std::get_if<ast::Pattern::Named>(&pattern))
            {
                auto& name = named->ident->ident;
                return {name, no_type};
            }
            else
            {
                auto typed = std::get<ast::Pattern::Typed>(pattern);
                assert(
                    std::holds_alternative<ast::Pattern::Named>(typed.pattern->pattern) &&
                    "Only named patterns are supported for typed variable declaration!");
                named = &std::get<ast::Pattern::Named>(typed.pattern->pattern);
                auto& name = named->ident->ident;
                assert(
                    std::holds_alternative<ast::Type::Var>(typed.type->type) &&
                    "Only var types are supported for typed variable declaration!");
                auto& type_name = std::get<ast::Type::Var>(typed.type->type).ident->ident;
                return {name, type_name};
            }
        }

        // We can get the name like this since a function must always have a name that only consists of one identifier
        const std::string& getFuncName(const ast::Pattern::Typed& typed_pattern)
        {
            auto& idents = std::get<ast::Pattern::LongIdent>(typed_pattern.pattern->pattern).ident;
            if (idents->idents.size() > 1)
                mlir::emitError(loc(typed_pattern), "Function names do not allow for nested identifiers!");
            return idents->idents.front()->ident;
        }

        // We can get the name like this since a function must always have a name that only consists of one identifier
        const std::string& getFuncName(const ast::Pattern::LongIdent& typed_pattern)
        {
            auto& idents = typed_pattern.ident;
            if (idents->idents.size() > 1)
                mlir::emitError(loc(typed_pattern), "Function names do not allow for nested identifiers!");
            return idents->idents.front()->ident;
        }

        std::tuple<mlir::StringRef, mlir::Type> getFunctionArg(const ast::Pattern& pattern)
        {
            if (auto paren = std::get_if<ast::Pattern::Paren>(&pattern.pattern))
                return getFunctionArg(*paren->pattern);
            if (auto typed = std::get_if<ast::Pattern::Typed>(&pattern.pattern))
            {
                if (auto named = std::get_if<ast::Pattern::Named>(&typed->pattern->pattern))
                {
                    return {
                        named->ident->ident,
                        mlir::fsharp::utils::getMLIRType(*typed->type, builder.getContext(), loc(*typed))
                    };
                }

                mlir::emitError(loc(pattern), "Invalid function arg name. Only non nested names allowed!");
                return {"", nullptr};
            }
            if (auto named = std::get_if<ast::Pattern::Named>(&pattern.pattern))
                return {named->ident->ident, builder.getNoneType()};

            mlir::emitError(loc(pattern), "Invalid function arg pattern.");
            return {"", nullptr};
        }

        std::tuple<mlir::SmallVector<mlir::StringRef, 4>, mlir::SmallVector<mlir::Type, 4>> getFunctionSignature(
            const ast::Pattern& pattern)
        {
            // If we are looking at a long ident pattern we only want to look at the patterns following the identifier
            if (auto long_ident = std::get_if<ast::Pattern::LongIdent>(&pattern.pattern))
            {
                mlir::SmallVector<mlir::StringRef, 4> arg_names;
                mlir::SmallVector<mlir::Type, 4> arg_types;
                for (auto& p : long_ident->patterns)
                {
                    auto [name, type] = getFunctionArg(*p);
                    arg_names.push_back(name);
                    arg_types.push_back(type);
                }
                return {arg_names, arg_types};
            }
            // If we are looking at a typed pattern for the function definition we only want to look at the enclosed pattern
            if (auto typed = std::get_if<ast::Pattern::Typed>(&pattern.pattern))
            {
                return getFunctionSignature(*typed->pattern);
            }

            mlir::emitError(loc(pattern), "Invalid function definition pattern.");
            return {};
        }

        mlir::fsharp::ClosureOp getClosureProto(const ast::Expression::Let& let)
        {
            mlir::StringRef func_name;
            mlir::Type return_type = mlir::NoneType::get(builder.getContext());
            // If the return-type of the function is specified we should save the return-type for later
            if (auto typed = std::get_if<ast::Pattern::Typed>(&let.args->pattern))
            {
                func_name = getFuncName(*typed);
                return_type = mlir::fsharp::utils::getMLIRType(*typed->type, builder.getContext(), loc(*typed));
            }
            else if (auto long_ident = std::get_if<ast::Pattern::LongIdent>(&let.args->pattern))
            {
                func_name = getFuncName(*long_ident);
            }
            else
            {
                mlir::emitError(loc(let), "Invalid function definition pattern.");
                return nullptr;
            }

            auto [sig_names, sig_types] = getFunctionSignature(*let.args);

            auto funcType = builder.getFunctionType(sig_types, return_type);
            auto function = builder.create<mlir::fsharp::ClosureOp>(loc(let), func_name, funcType);
            if (!function)
                return nullptr;

            mlir::Block& entryBlock = function.front();

            // Declare all the function arguments in the symbol table.
            for (const auto nameValue :
                 llvm::zip(sig_names, entryBlock.getArguments()))
            {
                if (failed(declare(std::get<0>(nameValue),
                                   std::get<1>(nameValue))))
                    return nullptr;
            }

            return function;
        }

        llvm::LogicalResult declareClosure(const ast::Expression::Let& let)
        {
            mlir::OpBuilder::InsertionGuard guard(builder);
            // Create a scope in the symbol table to hold variable declarations.
            llvm::ScopedHashTableScope<llvm::StringRef, mlir::Value> varScope(symbolTable);

            mlir::fsharp::ClosureOp closure = getClosureProto(let);
            if (let.isRecursive)
            {
                mlir::fsharp::utils::addOrUpdateAttrDictEntry(closure, "recursive", builder.getUnitAttr());
            }
            lastDeclaredFunction = closure;
            if (!closure)
                return llvm::failure();

            auto func_type = closure.getFunctionType();
            mlir::Block& entry_block = closure.front();

            // Set the insertion point in the builder to the beginning of the function
            // body, it will be used throughout the codegen to create operations in this
            // function.
            builder.setInsertionPointToStart(&entry_block);

            // Emit the body of the function.
            auto body_result = mlirGen(let.expressions);
            if (body_result.has_value() && body_result.value() == nullptr)
            {
                mlir::emitError(loc(let), "Invalid last statement!");
                return llvm::failure();
            }
            if (!llvm::isa<mlir::NoneType>(func_type.getResult(0)) && !body_result.has_value())
            {
                mlir::emitError(
                    loc(let), "Function does not return a value even though a return type was specified!");
                return llvm::failure();
            }

            // If the function returns a specified type, we need to set the return type of the function
            // But only if the last statement actually returns a value
            if (body_result.has_value())
                if (!mlir::isa<mlir::NoneType>(closure.getFunctionType().getResult(0)))
                    body_result.value().setType(closure.getFunctionType().getResult(0));

            builder.setInsertionPointToEnd(&entry_block);
            if (body_result.has_value())
            {
                builder.create<mlir::fsharp::ReturnOp>(body_result->getLoc(), body_result.value());
                closure.setFunctionType(mlir::FunctionType::get(
                        builder.getContext(),
                        closure.getFunctionType().getInputs(),
                        body_result->getType())
                );
            }
            else
            {
                builder.create<mlir::fsharp::ReturnOp>(loc(let));
                closure.setFunctionType(mlir::FunctionType::get(
                        builder.getContext(),
                        closure.getFunctionType().getInputs(),
                        {})
                );
            }


            return llvm::success();
        }


        mlir::Value generateVariable(const ast::Expression::Let& let)
        {
            auto [name, type] = getLetArg(let.args->pattern);
            if (let.expressions.empty())
            {
                mlir::emitError(loc(let.get_range()), "No initializer given to variable declaration!");
                return nullptr;
            }

            mlir::SmallVector<mlir::Value, 4> expressions;
            for (auto& expr : let.expressions)
            {
                auto value = mlirGen(*expr);
                if (value.has_value() && value.value() == nullptr)
                {
                    mlir::emitError(loc(let), "Invalid statement in let expression!");
                    return nullptr;
                }

                expressions.push_back(value.has_value() ? value.value() : nullptr);
            }

            auto value = expressions.back();

            if (type.get() != "")
            {
                genCast(builder, loc(let.get_range()), value,
                        mlir::fsharp::utils::getMLIRType(type.get(), builder.getContext()));
            }


            if (llvm::failed(declare(name.get(), value)))
                return nullptr;
            return value;
        }

        // Value is returned if the let expression defines a variable, LogicalResult is returned if it defines a function
        std::variant<mlir::Value, llvm::LogicalResult> mlirGen(const ast::Expression::Let& let)
        {
            // Check if this is a function definition
            if (std::holds_alternative<ast::Pattern::LongIdent>(let.args->pattern) ||
                // If the pattern is a typed pattern, for it to be a function definition the enclosed pattern must be a long ident
                (std::holds_alternative<ast::Pattern::Typed>(let.args->pattern) &&
                    std::holds_alternative<ast::Pattern::LongIdent>(
                        std::get<ast::Pattern::Typed>(let.args->pattern).pattern->pattern)))
            {
                return declareClosure(let);
            }
            return generateVariable(let);
        }

        mlir::Value mlirGen(const ast::Expression::Ident& ident)
        {
            if (auto variable = symbolTable.lookup(ident.ident->ident))
                return variable;

            mlir::emitError(loc(ident.get_range()),
                            fmt::format("error: unknown variable '{}'!", ident.ident->ident));
            return nullptr;
        }

        /// Emit a literal/constant array. It will be emitted as a flattened array of
        /// data in an Attribute attached to a `toy.constant` operation.
        /// See documentation on [Attributes](LangRef.md#attributes) for more details.
        /// Here is an excerpt:
        ///
        ///   Attributes are the mechanism for specifying constant data in MLIR in
        ///   places where a variable is never allowed [...]. They consist of a name
        ///   and a concrete attribute value. The set of expected attributes, their
        ///   structure, and their interpretation are all contextually dependent on
        ///   what they are attached to.
        ///
        /// Example, the source level statement:
        ///   let a = [1, 2, 3, 4, 5, 6];
        /// will be converted to:
        ///   %0 = "arith.constant"() {value: dense<tensor<6xi32>,
        ///     [1, 2, 3, 4, 5 ,6]>} : () -> tensor<6xi32>
        ///
        mlir::Value mlirGen(const ast::Constant& constant)
        {
            return getValue(constant);
        }

        mlir::Type getType(const ast::Constant& constant)
        {
            auto value = constant.value.value();
            return std::visit<mlir::Type>(::utils::overloaded{
                                              [&](const int32_t&)
                                              {
                                                  return builder.getIntegerType(32, true);
                                              },
                                              [&](const double_t&)
                                              {
                                                  return builder.getF64Type();
                                              },
                                              [&](const std::string& s)
                                              {
                                                  return mlir::RankedTensorType::get(
                                                      {static_cast<int64_t>(s.size() + 1)}, builder.getI8Type());
                                              },
                                              [&](const bool&)
                                              {
                                                  return builder.getI1Type();
                                              },
                                          }, value);
        }

        mlir::Value getValue(const ast::Constant& constant)
        {
            auto value = constant.value.value();
            auto type = getType(constant);
            return std::visit<mlir::Value>(::utils::overloaded{
                                               [&](const int32_t& i)
                                               {
                                                   return builder.create<mlir::fsharp::ConstantOp>(
                                                       loc(constant.get_range()), type,
                                                       builder.getSI32IntegerAttr(i));
                                               },
                                               [&](const double_t& f)
                                               {
                                                   return builder.create<mlir::fsharp::ConstantOp>(
                                                       loc(constant.get_range()), type, builder.getF64FloatAttr(f));
                                               },
                                               [&](const std::string& s)
                                               {
                                                   std::vector<char8_t> data{s.begin(), s.end()};
                                                   data.push_back('\0');
                                                   auto dataAttribute = mlir::DenseElementsAttr::get(
                                                       mlir::dyn_cast<mlir::ShapedType>(type),
                                                       llvm::ArrayRef(data));
                                                   auto tensor = builder.create<mlir::fsharp::ConstantOp>(
                                                       loc(constant.get_range()), type, dataAttribute);
                                                   return builder.create<mlir::tensor::CastOp>(loc(constant),
                                                       mlir::UnrankedTensorType::get(builder.getI8Type()),
                                                       tensor);
                                               },
                                               [&](const bool& b)
                                               {
                                                   return builder.create<mlir::fsharp::ConstantOp>(
                                                       loc(constant.get_range()), type,
                                                       builder.getIntegerAttr(builder.getI1Type(), b));
                                               },
                                           }, value);
        }
    };

    mlir::OwningOpRef<mlir::ModuleOp> MLIRGen::mlirGen(mlir::MLIRContext& context, std::string_view source,
                                                       std::string_view source_filename)
    {
        auto result = fsharpgrammar::Grammar::parse(source, true, false, true);
        return mlirGen(context, result, source_filename);
    }

    mlir::OwningOpRef<mlir::ModuleOp> MLIRGen::mlirGen(mlir::MLIRContext& context, std::unique_ptr<ast::Main>& ast,
                                                       std::string_view source_filename)
    {
        context.getOrLoadDialect<mlir::fsharp::FSharpDialect>();
        context.getOrLoadDialect<mlir::BuiltinDialect>();
        context.getOrLoadDialect<mlir::arith::ArithDialect>();
        context.getOrLoadDialect<mlir::LLVM::LLVMDialect>();
        context.getOrLoadDialect<mlir::func::FuncDialect>();
        context.getOrLoadDialect<mlir::bufferization::BufferizationDialect>();
        context.getOrLoadDialect<mlir::scf::SCFDialect>();
        context.getOrLoadDialect<mlir::cf::ControlFlowDialect>();

        mlir::DialectRegistry registry;
        mlir::arith::registerBufferizableOpInterfaceExternalModels(registry);
        mlir::tensor::registerBufferizableOpInterfaceExternalModels(registry);
        mlir::bufferization::func_ext::registerBufferizableOpInterfaceExternalModels(registry);
        mlir::scf::registerBufferizableOpInterfaceExternalModels(registry);
        mlir::registerConvertMemRefToLLVMInterface(registry);
        context.appendDialectRegistry(registry);

        context.getOrLoadDialect<mlir::fsharp::FSharpDialect>();

        return MLIRGenImpl(context, source_filename).mlirGen(*ast);
    }
}
