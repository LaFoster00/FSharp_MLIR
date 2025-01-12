//
// Created by lasse on 1/7/25.
//
#include "compiler/ASTToMLIR.h"
#include "compiler/FSharpDialect.h"
#include <span>

#include <ast/ASTNode.h>
#include <ast/Range.h>
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
#include "mlir/Dialect/MemRef/TransformOps/MemRefTransformOps.h"

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
        llvm::LogicalResult declare(llvm::StringRef var, mlir::Value value)
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

        mlir::Value mlirGen(const ast::ModuleDeclaration::Expression& expression)
        {
            return mlirGen(*expression.expression);
        }

        mlir::Value mlirGen(const ast::Expression& expression)
        {
            if (std::holds_alternative<ast::Expression::Append>(expression.expression))
                return mlirGen(std::get<ast::Expression::Append>(expression.expression));
            if (std::holds_alternative<ast::Expression::Constant>(expression.expression))
                return mlirGen(std::get<ast::Expression::Constant>(expression.expression));

            throw std::runtime_error("Expression not supported!");
        }

        std::span<const ast::ast_ptr<ast::Expression>> getFunctionArgs(const ast::Expression::Append& append)
        {
            return {std::next(append.expressions.begin()), append.expressions.end()};
        }

        mlir::Value mlirGen(const ast::Expression::Append& append)
        {
            if (append.isFunctionCall)
            {
                return declareFunctionCall(append);
            }
            else
            {
                return mlirGen(*append.expressions.front());
            }
            return nullptr;
        }

        mlir::Value declareFunctionCall(const ast::Expression::Append& append)
        {
            std::string func_name;
            if (std::holds_alternative<ast::Expression::Ident>(append.expressions.front()->expression))
                func_name = std::get<ast::Expression::Ident>(append.expressions.front()->expression).ident->ident;
            else if (std::holds_alternative<ast::Expression::LongIdent>(append.expressions.front()->expression))
                func_name = std::get<ast::Expression::LongIdent>(append.expressions.front()->expression).longIdent->
                    get_as_string();

            auto location = loc(append.get_range());

            // Codegen the operands first
            llvm::SmallVector<mlir::Value, 4> operands;
            for (auto& expr : getFunctionArgs(append))
            {
                const auto arg = mlirGen(*expr);
                if (!arg)
                    return nullptr;
                operands.push_back(arg);
            }

            return builder.create<mlir::fsharp::GenericCallOp>(location, func_name, operands);
        }

        llvm::LogicalResult declarePrintExpr(const ast::Expression::Append& append)
        {
            // Get the printf formating argument
            auto arg = mlirGen(*(*std::next(append.expressions.begin())));
            if (!arg)
                return mlir::failure();

            builder.create<mlir::fsharp::PrintOp>(loc(append.get_range()), arg);
            return mlir::success();
        }

        mlir::Value mlirGen(const ast::Expression::Constant& constant)
        {
            auto value = getValue(constant);
            return builder.create<mlir::arith::ConstantOp>(
                loc(constant.get_range()),
                value.getType(),
                value
            );
        }

        mlir::TypedAttr getValue(const ast::Expression::Constant& constant)
        {
            auto& value = constant.constant->value.value();
            return std::visit<mlir::TypedAttr>(utils::overloaded{
                                                   [&](const int32_t i) { return builder.getI32IntegerAttr(i); },
                                                   [&](const float_t f) { return builder.getF32FloatAttr(f); },
                                                   [&](const std::string& s)
                                                   {
                                                       auto string_attr = mlir::fsharp::StringAttr::get(builder.getContext(), s);
                                                       return string_attr;
                                                   },
                                                   [&](const char8_t c) { return builder.getI8IntegerAttr(c); },
                                                   [&](const bool b) { return builder.getBoolAttr(b); },
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
        context.getOrLoadDialect<mlir::fsharp::FSharpDialect>();

        return MLIRGenImpl(context, source_filename).mlirGen(*ast);
    }
}
