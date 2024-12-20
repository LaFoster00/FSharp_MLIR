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
            if (result.has_value())
                expressions.push_back(ast::any_cast<Expression>(result, context));
        }

        if (expressions.size() > 1)
            return make_ast<Expression>(
                Expression::Sequential(
                    std::move(expressions),
                    false,
                    Range::create(context))
            );
        else if (expressions.size() == 1)
            return expressions.front();
        else
            return make_ast<Expression>(PlaceholderNodeAlternative("Sequential Expr"));
    }

    std::any AstBuilder::visitExpression(FSharpParser::ExpressionContext* ctx)
    {
        if (ctx->assignment_expr())
            return ctx->assignment_expr()->accept(this);
        if (ctx->non_assigment_expr())
            return ctx->non_assigment_expr()->accept(this);

        throw std::runtime_error("Invalid expression");
    }

    std::any AstBuilder::visitNon_assigment_expr(FSharpParser::Non_assigment_exprContext* context)
    {
        std::vector<ast_ptr<Expression>> expressions;
        for (auto tuple_expr : context->tuple_expr())
        {
            auto result = tuple_expr->accept(this);
            if (result.has_value())
                expressions.push_back(ast::any_cast<Expression>(result, context));
        }
        if (expressions.size() == 2)
            return make_ast<Expression>(
                Expression::Append(
                    std::move(expressions[0]),
                    std::move(expressions[1]),
                    Range::create(context))
            );
        else if (expressions.size() == 1)
            return expressions[0];
        else
            return {};
    }

    std::any AstBuilder::visitTuple_expr(FSharpParser::Tuple_exprContext* context)
    {
        return make_ast<Expression>(
            PlaceholderNodeAlternative("Tuple Expr")
        );
    }

    std::any AstBuilder::visitAssignment_expr(FSharpParser::Assignment_exprContext* context)
    {
        switch (context->getRuleIndex())
        {
        case 0:
            return context->let_stmt()->accept(this);
        default:
            return {};
        }
    }
} // fsharpgrammar
