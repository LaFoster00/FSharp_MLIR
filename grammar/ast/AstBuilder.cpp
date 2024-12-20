//
// Created by lasse on 12/18/24.
//

#include "AstBuilder.h"

#include "ASTNode.h"
#include "ASTHelper.h"

namespace fsharpgrammar
{
    std::any AstBuilder::visitMain(FSharpParser::MainContext* ctx)
    {
        std::vector<ast_ptr<ModuleOrNamespace>> anon_modules;
        for (auto module_or_namespace : ctx->module_or_namespace())
        {
            std::any module_result = module_or_namespace->accept(this);
            if (module_result.has_value())
            {
                anon_modules.push_back(ast::any_cast<ModuleOrNamespace>(module_result, ctx));
            }
        }
        return make_ast<Main>(std::move(anon_modules), Range::create(ctx));
    }

    std::vector<ast_ptr<ModuleDeclaration>> get_module_declarations(
        std::vector<FSharpParser::Module_declContext*> decls,
        antlr4::ParserRuleContext* context,
        FSharpParserVisitor* visitor)

    {
        std::vector<ast_ptr<ModuleDeclaration>> module_declarations;
        for (auto module_decl : decls)
        {
            std::any module_result = module_decl->accept(visitor);
            if (module_result.has_value())
            {
                module_declarations.push_back(ast::any_cast<ModuleDeclaration>(module_result, context));
            }
        }
        return module_declarations;
    }

    std::any AstBuilder::visitAnonmodule(FSharpParser::AnonmoduleContext* context)
    {
        return make_ast<ModuleOrNamespace>(
            ModuleOrNamespace::Type::AnonymousModule,
            std::optional<std::string>{},
            get_module_declarations(context->module_decl(), context, this),
            Range::create(context));
    }

    std::any AstBuilder::visitNamedmodule(FSharpParser::NamedmoduleContext* context)
    {
        return make_ast<ModuleOrNamespace>(
            ModuleOrNamespace::Type::NamedModule,
            ast::to_string(context->long_ident()),
            get_module_declarations(context->module_decl(), context, this),
            Range::create(context));
    }

    std::any AstBuilder::visitNamespace(FSharpParser::NamespaceContext* ctx)
    {
        return make_ast<ModuleOrNamespace>(
            ModuleOrNamespace::Type::Namespace,
            ast::to_string(ctx->long_ident()),
            get_module_declarations(ctx->module_decl(), ctx, this),
            Range::create(ctx));
    }

    std::any AstBuilder::visitNested_module(FSharpParser::Nested_moduleContext* context)
    {
        std::vector<ast_ptr<ModuleDeclaration>> module_declarations;
        for (auto module_decl : context->module_decl())
        {
            auto result = module_decl->accept(this);
            module_declarations.push_back(ast::any_cast<ModuleDeclaration>(result, context));
        }

        ModuleDeclaration::NestedModule nested_module(
            ast::to_string(context->long_ident()),
            std::move(module_declarations),
            Range::create(context));

        return make_ast<ModuleDeclaration>(
            std::move(nested_module));
    }

    std::any AstBuilder::visitExpression_stmt(FSharpParser::Expression_stmtContext* context)
    {
        auto expression = context->sequential_stmt()->accept(this);
        ast_ptr<Expression> result;
        if (expression.has_value())
            result = ast::any_cast<Expression>(expression, context);

        return make_ast<ModuleDeclaration>(
            ModuleDeclaration::Expression(
                std::move(result),
                Range::create(context)
            )
        );
    }

    std::any AstBuilder::visitOpen_stmt(FSharpParser::Open_stmtContext* context)
    {
        return make_ast<ModuleDeclaration>(
            ModuleDeclaration::Open(
                ast::to_string(context->long_ident()),
                Range::create(context)));
    }

    std::any AstBuilder::visitSequential_stmt(FSharpParser::Sequential_stmtContext* context)
    {
        std::vector<ast_ptr<Expression>> expressions;
        for (auto expr : context->expression())
        {
            auto result = expr->accept(this);
            expressions.push_back(ast::any_cast<Expression>(result, context));
        }

        return make_ast<Expression>(
            Expression::Sequential(
                std::move(expressions),
                false,
                Range::create(context))
        );
    }

    std::any AstBuilder::visitExpression(FSharpParser::ExpressionContext* ctx)
    {
        return make_ast<Expression>(Expression::Sequential(std::vector<ast_ptr<Expression>>{}, true, Range::create(ctx)));
    }
} // fsharpgrammar
