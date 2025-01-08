//
// Created by lasse on 1/7/25.
//
#include "ASTToMLIR.h"

#include <ast/ASTNode.h>
#include <ast/Range.h>

#include "Grammar.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Verifier.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/ScopedHashTable.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"

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
                return nullptr;
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
            llvm::ScopedHashTableScope<llvm::StringRef, mlir::Value> varScope(symbolTable);

            auto previous_insertion_block = builder.getInsertionBlock();
            auto m = builder.create<
                mlir::ModuleOp>(loc(nested_module.get_range()), nested_module.name->get_as_string());

            // Move into new module
            builder.setInsertionPointToStart(m.getBody());

            // Emit the body of the module
            for (auto module_decl : nested_module.moduleDecls)
            {
                if (std::holds_alternative<ast::ModuleDeclaration::NestedModule>(module_decl->declaration))
                {
                    mlirGen(std::get<ast::ModuleDeclaration::NestedModule>(module_decl->declaration));
                }
                else if (std::holds_alternative<ast::ModuleDeclaration::Expression>(module_decl->declaration))
                {
                    mlirGen(*std::get<ast::ModuleDeclaration::Expression>(module_decl->declaration).expression);
                }
                else
                {
                    throw std::runtime_error("Open statements not supported!");
                }
            }

            // Move back to outer scope
            builder.setInsertionPointToEnd(previous_insertion_block);

            return m;
        }

        mlir::Value mlirGen(const ast::ModuleDeclaration::Expression &expression)
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

        mlir::Value mlirGen(const ast::Expression::Append& append)
        {
            if (append.isFunctionCall)
            {
                if (std::holds_alternative<ast::Expression::Ident>(append.expressions.front()->expression))
                {

                }
            }

        }

        mlir::Value mlirGen(const ast::Expression::Constant& constant)
        {
            auto type = getType(*constant.constant);
            auto value = constant.constant->value.value();
            if (std::holds_alternative<int32_t>(value))
            {
                return builder.create<mlir::arith::ConstantIntOp>(loc(constant.get_range()), std::get<int32_t>(value), type);
            }
        }

        mlir::Type getType(const ast::Constant& constant)
        {
            if (std::holds_alternative<int32_t>(*constant.value))
                return builder.getI32Type();
            else
                throw std::runtime_error("Constant not supported!");
        }
    };


    mlir::OwningOpRef<mlir::ModuleOp> MLIRGen::mlirGen(mlir::MLIRContext& context, std::string_view source,
                                                       std::string_view source_filename)
    {
        context.getOrLoadDialect<mlir::BuiltinDialect>();
        context.getOrLoadDialect<mlir::arith::ArithDialect>();

        auto result = fsharpgrammar::Grammar::parse(source, true, false, true);
        return MLIRGenImpl(context, source_filename).mlirGen(*result);
    }
}
