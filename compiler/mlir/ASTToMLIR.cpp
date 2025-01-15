//
// Created by lasse on 1/7/25.
//
#include "compiler/ASTToMLIR.h"
#include "compiler/FSharpDialect.h"
#include <span>
#include <fmt/color.h>

#include <ast/ASTNode.h>
#include <ast/Range.h>
#include <mlir/InitAllDialects.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>

#include "Grammar.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Verifier.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/ScopedHashTable.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/FunctionCallUtils.h"

namespace fsharpgrammar::compiler
{
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
            llvm::ScopedHashTableScope<llvm::StringRef, mlir::Value> varScope(symbolTable);

            for (auto& f : main_ast.modules_or_namespaces)
                mlirGen(*f);

            // Verify the module after we have finished constructing it, this will check
            // the structural properties of the IR and invoke any specific verifiers we
            // have on the Toy operations.
            if (failed(mlir::verify(fileModule)))
            {
                fileModule.emitError("module verification error");
                //return nullptr;
            }

            return fileModule;
        }

    private:
        std::string_view filename;

        /// A "module" matches a fsharp source file: containing a list of functions.
        mlir::ModuleOp fileModule;

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
            return mlir::FileLineColLoc::get(
                builder.getStringAttr(filename),
                range.start_line(),
                range.start_column());
        }

        /// Declare a variable in the current scope, return success if the variable
        /// wasn't declared yet.
        llvm::LogicalResult declare(const llvm::StringRef var, mlir::Value value)
        {
            if (symbolTable.count(var))
                return mlir::failure();
            symbolTable.insert(var, value);
            return mlir::success();
        }

    private:
        mlir::ModuleOp mlirGen(const ast::ModuleOrNamespace& module_or_namespace)
        {
            builder.setInsertionPointToEnd(fileModule.getBody());
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
                m = builder.create<mlir::ModuleOp>(loc(module_or_namespace.range));
                break;
            }
            builder.setInsertionPointToEnd(m.getBody());

            for (auto& module_decl : module_or_namespace.moduleDecls)
            {
                std::visit([&](auto& obj) { mlirGen(obj); },
                           module_decl->declaration);
            }

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

        std::optional<mlir::Value> mlirGen(const ast::Expression& expression)
        {
            if (std::holds_alternative<ast::Expression::Append>(expression.expression))
            {
                if (auto result = mlirGen(std::get<ast::Expression::Append>(expression.expression));
                    std::holds_alternative<mlir::Value>(result))
                    return std::get<mlir::Value>(result);
                return {};
            }
            if (std::holds_alternative<ast::Expression::Constant>(expression.expression))
                return mlirGen(std::get<ast::Expression::Constant>(expression.expression));
            if (std::holds_alternative<ast::Expression::Let>(expression.expression))
                return mlirGen(std::get<ast::Expression::Let>(expression.expression));
            if (std::holds_alternative<ast::Expression::Ident>(expression.expression))
                return mlirGen(std::get<ast::Expression::Ident>(expression.expression));

            mlir::emitError(loc(expression.get_range()), "Expression not supported!");
        }

        std::span<const ast::ast_ptr<ast::Expression>> getFunctionArgs(const ast::Expression::Append& append)
        {
            return {std::next(append.expressions.begin()), append.expressions.end()};
        }

        std::optional<llvm::SmallVector<mlir::Value, 4>> getFunctionArgValues(const ast::Expression::Append& append)
        {
            // Codegen the operands first
            llvm::SmallVector<mlir::Value, 4> operands;
            for (auto& expr : getFunctionArgs(append))
            {
                const auto arg = mlirGen(*expr);
                if (!arg)
                    return {};
                if (!arg.has_value())
                {
                    fmt::print(fmt::fg(fmt::color::orange_red), "Function argument value does not return a value! {}",
                               utils::to_string(expr->get_range()));
                    operands.push_back(nullptr);
                    continue;
                };
                operands.push_back(arg.value());
            }
            return operands;
        }

        std::variant<mlir::Value, llvm::LogicalResult> mlirGen(const ast::Expression::Append& append)
        {
            if (append.isFunctionCall)
            {
                std::string func_name;
                if (std::holds_alternative<ast::Expression::Ident>(append.expressions.front()->expression))
                    func_name = std::get<ast::Expression::Ident>(append.expressions.front()->expression).ident->ident;
                else if (std::holds_alternative<ast::Expression::LongIdent>(append.expressions.front()->expression))
                    func_name = std::get<ast::Expression::LongIdent>(append.expressions.front()->expression).longIdent->
                        get_as_string();

                if (func_name == "print" || func_name == "printf" || func_name == "printfn")
                    return generatePrint(append);

                return declareFunctionCall(append, func_name);
            }
            else
            {
                return mlirGen(*append.expressions.front()).value();
            }
            return mlir::Value(nullptr);
        }

        mlir::Value declareFunctionCall(const ast::Expression::Append& append, const std::string& func_name)
        {
            auto location = loc(append.get_range());

            auto args = getFunctionArgValues(append);
            if (!args.has_value())
                return nullptr;
            return builder.create<mlir::func::CallOp>(location, mlir::StringRef(func_name),
                                                      mlir::ValueRange(args.value()))->getResult(0);
        }

        llvm::LogicalResult generatePrint(const ast::Expression::Append& append)
        {
            auto args = getFunctionArgValues(append);
            if (!args.has_value())
                return llvm::failure();
            if (mlir::cast<mlir::ShapedType>(args.value().front().getType()).getElementType() == builder.getI8Type())
                builder.create<mlir::fsharp::PrintStringOp>(loc(append.get_range()), mlir::ValueRange(args.value()));
            else
                builder.create<mlir::fsharp::PrintArrayOp>(loc(append.get_range()), mlir::ValueRange(args.value()));
            return llvm::success();
        }

        mlir::Value mlirGen(const ast::Expression::Constant& constant)
        {
            return mlirGen(*constant.constant);
        }

        std::tuple<std::reference_wrapper<const std::string>, std::reference_wrapper<const std::string>> getLetArg(
            const ast::Pattern::PatternType& pattern)
        {
            static const std::string no_type = "";
            if (std::holds_alternative<ast::Pattern::Named>(pattern))
            {
                auto named = std::get<ast::Pattern::Named>(pattern);
                auto& name = named.ident->ident;
                return {name, no_type};
            }
            else
            {
                auto typed = std::get<ast::Pattern::Typed>(pattern);
                assert(
                    std::holds_alternative<ast::Pattern::Named>(typed.pattern->pattern) &&
                    "Only named patterns are supported for typed variable declaration!");
                auto named = std::get<ast::Pattern::Named>(typed.pattern->pattern);
                auto& name = named.ident->ident;
                assert(
                    std::holds_alternative<ast::Type::Var>(typed.type->type) &&
                    "Only var types are supported for typed variable declaration!");
                auto& type_name = std::get<ast::Type::Var>(typed.type->type).ident->ident;
                return {name, type_name};
            }
        }

        mlir::Type getMLIRType(const std::string& type_name)
        {
            if (type_name == "int")
                return mlir::RankedTensorType::get({1}, builder.getI32Type());
            if (type_name == "float")
                return mlir::RankedTensorType::get({1}, builder.getF32Type());
            if (type_name == "bool")
                return mlir::RankedTensorType::get({1}, builder.getI8Type());
            if (type_name == "string")
                return mlir::UnrankedTensorType::get(builder.getI8Type());
            assert(false && "Type not supported!");
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
                expressions.push_back(value.has_value() ? value.value() : nullptr);
            }

            auto value = expressions.back();

            if (type.get() != "")
            {
                if (value.getType() != getMLIRType(type))
                {
                    if (!mlir::tensor::CastOp::areCastCompatible(value.getType(), getMLIRType(type)))
                    {
                        mlir::emitError(loc(let.get_range()),
                                        fmt::format(
                                            "Variable declaration not compatible with specified variable type {}!",
                                            type.get()));
                        return nullptr;
                    }
                    value = builder.create<mlir::tensor::CastOp>(loc(let.get_range()), getMLIRType(type),
                                                                 expressions.back());
                }
            }


            if (llvm::failed(declare(name.get(), value)))
                return nullptr;
            return value;
        }

        mlir::Value mlirGen(const ast::Expression::Let& let)
        {
            // Check if this is a function definition
            if (std::holds_alternative<ast::Pattern::LongIdent>(let.args->pattern))
            {
                mlir::emitError(loc(let.get_range()), "Function definitions not supported!");
            }
            return generateVariable(let);
        }

        mlir::Value mlirGen(const ast::Expression::Ident& ident)
        {
            if (auto variable = symbolTable.lookup(ident.ident->ident))
                return variable;

            mlir::emitError(loc(ident.get_range()), fmt::format("error: unknown variable '{}'!", ident.ident->ident));
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

        mlir::ShapedType getType(const ast::Constant& constant)
        {
            auto value = constant.value.value();
            return std::visit<mlir::ShapedType>(utils::overloaded{
                                                    [&](const int32_t&)
                                                    {
                                                        return mlir::RankedTensorType::get({1}, builder.getI32Type());
                                                    },
                                                    [&](const float_t&)
                                                    {
                                                        return mlir::RankedTensorType::get({1}, builder.getF32Type());
                                                    },
                                                    [&](const std::string& s)
                                                    {
                                                        return mlir::RankedTensorType::get(
                                                            {static_cast<int64_t>(s.size() + 1)}, builder.getI8Type());
                                                    },
                                                    [&](const bool&)
                                                    {
                                                        return mlir::RankedTensorType::get({1}, builder.getI8Type());
                                                    },
                                                }, value);
        }

        mlir::Value getValue(const ast::Constant& constant)
        {
            auto value = constant.value.value();
            auto type = getType(constant);
            return std::visit<mlir::Value>(utils::overloaded{
                                               [&](const int32_t& i)
                                               {
                                                   const std::vector data = {i};
                                                   auto dataAttribute = mlir::DenseElementsAttr::get(
                                                       type, llvm::ArrayRef(data));
                                                   return builder.create<mlir::arith::ConstantOp>(
                                                       loc(constant.get_range()), type, dataAttribute);
                                               },
                                               [&](const float_t& f)
                                               {
                                                   const std::vector data = {f};
                                                   auto dataAttribute = mlir::DenseElementsAttr::get(
                                                       type, llvm::ArrayRef(data));
                                                   return builder.create<mlir::arith::ConstantOp>(
                                                       loc(constant.get_range()), type, dataAttribute);
                                               },
                                               [&](const std::string& s)
                                               {
                                                   std::vector<char8_t> data{s.begin(), s.end()};
                                                   data.push_back('\0');
                                                   auto dataAttribute = mlir::DenseElementsAttr::get(
                                                       type, llvm::ArrayRef(data));
                                                   return builder.create<mlir::arith::ConstantOp>(
                                                       loc(constant.get_range()), type, dataAttribute);
                                               },
                                               [&](const bool& b)
                                               {
                                                   const std::vector<int8_t> data = {b};
                                                   auto dataAttribute = mlir::DenseElementsAttr::get(
                                                       type, llvm::ArrayRef(data));
                                                   return builder.create<mlir::arith::ConstantOp>(
                                                       loc(constant.get_range()), type, dataAttribute);
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
        context.getOrLoadDialect<mlir::BuiltinDialect>();
        context.getOrLoadDialect<mlir::arith::ArithDialect>();
        context.getOrLoadDialect<mlir::LLVM::LLVMDialect>();
        context.getOrLoadDialect<mlir::func::FuncDialect>();
        context.getOrLoadDialect<mlir::bufferization::BufferizationDialect>();

        mlir::DialectRegistry registry;
        mlir::arith::registerBufferizableOpInterfaceExternalModels(registry);
        mlir::tensor::registerBufferizableOpInterfaceExternalModels(registry);
        context.appendDialectRegistry(registry);

        context.getOrLoadDialect<mlir::fsharp::FSharpDialect>();

        return MLIRGenImpl(context, source_filename).mlirGen(*ast);
    }
}
