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
        return context->app_expr()->accept(this);
    }

    std::any AstBuilder::visitApp_expr(FSharpParser::App_exprContext* context)
    {
        std::vector<ast_ptr<Expression>> expressions;
        for (auto tuple_expr : context->tuple_expr())
        {
            auto result = tuple_expr->accept(this);
            if (result.has_value())
                expressions.push_back(ast::any_cast<Expression>(result, context));
        }
        if (expressions.size() > 1)
            return make_ast<Expression>(
                Expression::Append(
                    std::move(expressions),
                    Range::create(context))
            );
        else if (expressions.size() == 1)
            return expressions[0];
        else
            return make_ast<Expression>(PlaceholderNodeAlternative("App Expr"));
    }

    std::any AstBuilder::visitTuple_expr(FSharpParser::Tuple_exprContext* context)
    {
        std::vector<ast_ptr<Expression>> expressions;
        for (auto expr : context->or_expr())
        {
            std::any result = expr->accept(this);
            if (result.has_value())
                expressions.push_back(ast::any_cast<Expression>(result, context));
        }
        if (expressions.size() > 1)
            return make_ast<Expression>(
                    Expression::Tuple(
                        std::move(expressions),
                        Range::create(context))
                );
        else if (expressions.size() == 1)
            return expressions[0];
        else
            return make_ast<Expression>(PlaceholderNodeAlternative("Implement Or Expr"));
    }

    std::any AstBuilder::visitOr_expr(FSharpParser::Or_exprContext* context)
    {
        if (context->and_expr().size() > 2)
            return make_ast<Expression>(PlaceholderNodeAlternative("Or Expr"));
        else
            return context->and_expr().front()->accept(this);
    }

    std::any AstBuilder::visitAnd_expr(FSharpParser::And_exprContext* context)
    {
        if (context->equality_expr().size() > 2)
            return make_ast<Expression>(PlaceholderNodeAlternative("And Expr"));
        else
            return context->equality_expr().front()->accept(this);
    }

    std::any AstBuilder::visitEquality_expr(FSharpParser::Equality_exprContext* context)
    {
        if (context->relation_expr().size() > 2)
            return make_ast<Expression>(PlaceholderNodeAlternative("Equality Expr"));
        else
            return context->relation_expr().front()->accept(this);
    }

    std::any AstBuilder::visitRelation_expr(FSharpParser::Relation_exprContext* context)
    {
        if (context->additive_expr().size() > 2)
            return make_ast<Expression>(PlaceholderNodeAlternative("Relation Expr"));
        else
            return context->additive_expr().front()->accept(this);
    }

    std::any AstBuilder::visitAdditive_expr(FSharpParser::Additive_exprContext* context)
    {
        if (context->multiplicative_expr().size() > 2)
            return make_ast<Expression>(PlaceholderNodeAlternative("Additive Expr"));
        else
            return context->multiplicative_expr().front()->accept(this);
    }

    std::any AstBuilder::visitMultiplicative_expr(FSharpParser::Multiplicative_exprContext* context)
    {
        if (context->dot_get_expr().size() > 2)
            return make_ast<Expression>(PlaceholderNodeAlternative("Multiplicative Expr"));
        else
            return context->dot_get_expr().front()->accept(this);
    }

    std::any AstBuilder::visitDot_get_expr(FSharpParser::Dot_get_exprContext* context)
    {
        return make_ast<Expression>(PlaceholderNodeAlternative("Dot Get Expr"));
    }

    std::any AstBuilder::visitAssignment_expr(FSharpParser::Assignment_exprContext* context)
    {
        switch (context->getRuleIndex())
        {
        case 0:
            return context->let_stmt()->accept(this);
        default:
            return make_ast<Expression>(PlaceholderNodeAlternative("Assignment Expr"));
        }
    }
} // fsharpgrammar
