//
// Created by lasse on 1/7/25.
//

#include "compiler/ASTToMLIR.h"

#include <chrono>
#include <iostream>
#include <thread>
#include <ast/ASTNode.h>

#include <gtest/gtest.h>

#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/Target/LLVMIR/ModuleImport.h"

using namespace fsharpgrammar::compiler;
using namespace std::chrono_literals;

mlir::OwningOpRef<mlir::ModuleOp> generate_mlir(std::string_view source,
                                                mlir::MLIRContext& context)
{
    return MLIRGen::mlirGen(
        context,
        source
    );
}

#define GENERATE_AND_DUMP_MLIR(source) \
    mlir::MLIRContext context; \
    auto result = generate_mlir(source, context); \
    std::cout << std::flush; \
    std::this_thread::sleep_for(100ms); \
    result->dump()

TEST(HelloWorld, BasicAssertion)
{
    GENERATE_AND_DUMP_MLIR(
        R"(
printfn "Hello, World!"
)"
    );
}

TEST(SimpleNamedModule, BasicAssertion)
{
    GENERATE_AND_DUMP_MLIR(
        R"(
module outer

printfn 1
)"
    );
}

TEST(SimpleNestedModule, BasicAssertion)
{
    GENERATE_AND_DUMP_MLIR(
        R"(
module outer
printfn 1
module nested =
    printfn 2
)"
    );
}

TEST(MultiNestedModule, BasicAssertion)
{
    GENERATE_AND_DUMP_MLIR(
        R"(
module outer
printfn 1
module nested =
    printfn 2
    module nested_nested =
        printfn 3
module nested2 =
    printfn 4
)"
    );
}

TEST(AritTest, BasicAssertion)
{
    GENERATE_AND_DUMP_MLIR(
        R"(
let a = 1 + 2
print a
)"
    );
}

TEST(SimpleAdd, BasicAssertion)
{
    using namespace fsharpgrammar;
    using namespace fsharpgrammar::ast;
    mlir::MLIRContext context;

    std::vector<ast_ptr<Expression>> add_expressions;
    add_expressions.push_back(
        make_ast<Expression>(Expression::Constant(
                make_ast<ast::Constant>(2, Range::create(0, 0)))
        )
    );
    add_expressions.push_back(
        make_ast<Expression>(Expression::Constant(
                make_ast<ast::Constant>(1, Range::create(0, 0)))
        )
    );

    std::vector<ast_ptr<ModuleDeclaration>> module_declarations;
    module_declarations.emplace_back(make_ast<ModuleDeclaration>(
            ModuleDeclaration::Expression(
                make_ast<Expression>(Expression::OP(
                        std::move(add_expressions),
                        Expression::OP::Type::ARITHMETIC,
                        std::vector{Expression::OP::ArithmeticType::ADD},
                        Range::create(0, 0, 0, 0)
                    )
                ),
                Range::create(0, 0, 0, 0)
            )
        )
    );

    auto module_or_namespace = make_ast<ModuleOrNamespace>(
        ModuleOrNamespace::Type::AnonymousModule,
        std::optional<ast_ptr<LongIdent>>{},
        std::move(module_declarations),
        Range::create(0, 0, 0, 0)
    );
    std::vector module_or_namespaces{module_or_namespace};

    auto main = std::make_unique<Main>(std::move(module_or_namespaces), Range::create(0, 0));

    std::cout << utils::to_string(*main) << std::endl;

    auto result = MLIRGen::mlirGen(
        context,
        main
    );
    std::cout << std::flush;
    std::this_thread::sleep_for(100ms);
    result->dump();
}
