//
// Created by lasse on 12/18/24.
//

#include "AstBuilder.h"

#include "ASTNode.h"
#include "ASTHelper.h"
#include "utils/FunctionTimer.h"

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
        const std::vector<FSharpParser::Module_declContext*>& decls,
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
            std::optional<ast_ptr<LongIdent>>{},
            get_module_declarations(context->module_decl(), context, this),
            Range::create(context));
    }

    std::any AstBuilder::visitNamedmodule(FSharpParser::NamedmoduleContext* context)
    {
        return make_ast<ModuleOrNamespace>(
            ModuleOrNamespace::Type::NamedModule,
            ast::any_cast<LongIdent>(context->long_ident()->accept(this), context),
            get_module_declarations(context->module_decl(), context, this),
            Range::create(context));
    }

    std::any AstBuilder::visitNamespace(FSharpParser::NamespaceContext* ctx)
    {
        return make_ast<ModuleOrNamespace>(
            ModuleOrNamespace::Type::Namespace,
            ast::any_cast<LongIdent>(ctx->long_ident()->accept(this), ctx),
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
            ast::any_cast<LongIdent>(context->long_ident()->accept(this), context),
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
                ast::any_cast<LongIdent>(context->long_ident()->accept(this), context),
                Range::create(context)));
    }

    std::any AstBuilder::visitMultiline_body(FSharpParser::Multiline_bodyContext* context)
    {
        std::vector<ast_ptr<Expression>> expressions;
        for (auto sequential_stmt : context->sequential_stmt())
        {
            if (!sequential_stmt->expression().empty())
                expressions.push_back(ast::any_cast<Expression>(sequential_stmt->accept(this), context));
        }
        return expressions;
    }

    std::any AstBuilder::visitSingle_line_body(FSharpParser::Single_line_bodyContext* context)
    {
        std::vector<ast_ptr<Expression>> expressions;
        expressions.push_back(ast::any_cast<Expression>(context->inline_sequential_stmt()->accept(this), context));
        return expressions;
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

        return expressions.front();
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
        return context->tuple_expr()->accept(this);
    }

    std::any AstBuilder::visitApp_expr(FSharpParser::App_exprContext* context)
    {
        std::vector<ast_ptr<Expression>> expressions;
        for (auto tuple_expr : context->or_expr())
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

        return expressions[0];
    }

    std::any AstBuilder::visitTuple_expr(FSharpParser::Tuple_exprContext* context)
    {
        std::vector<ast_ptr<Expression>> expressions;
        for (auto expr : context->app_expr())
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

        return expressions[0];
    }


    std::any AstBuilder::visitOr_expr(FSharpParser::Or_exprContext* context)
    {
        std::vector<ast_ptr<Expression>> results;
        results.reserve(context->children.size() / 2 + 1);

        std::vector<Expression::OP::LogicalType> operators;
        operators.reserve(context->children.size() / 2);

        for (const auto child : context->children)
        {
            if (const auto and_expr = dynamic_cast<decltype(context->and_expr(0))>(child))
            {
                if (auto result = and_expr->accept(this);
                    result.has_value())
                    results.push_back(ast::any_cast<Expression>(result, context));
                continue;
            }

            operators.push_back(Expression::OP::LogicalType::OR);
        }


        if (context->and_expr().size() > 1)
            return make_ast<Expression>(
                Expression::OP(
                    std::move(results),
                    Expression::OP::Type::LOGICAL,
                    std::move(operators),
                    Range::create(context))
            );
        else
            return context->and_expr().front()->accept(this);
    }

    std::any AstBuilder::visitAnd_expr(FSharpParser::And_exprContext* context)
    {
        std::vector<ast_ptr<Expression>> results;
        results.reserve(context->children.size() / 2 + 1);

        std::vector<Expression::OP::LogicalType> operators;
        operators.reserve(context->children.size() / 2);

        for (const auto child : context->children)
        {
            if (const auto expression = dynamic_cast<decltype(context->equality_expr(0))>(child))
            {
                if (auto result = expression->accept(this);
                    result.has_value())
                    results.push_back(ast::any_cast<Expression>(result, context));
                continue;
            }

            operators.push_back(Expression::OP::LogicalType::AND);
        }

        if (context->equality_expr().size() > 1)
            return make_ast<Expression>(
                Expression::OP(
                    std::move(results),
                    Expression::OP::Type::LOGICAL,
                    std::move(operators),
                    Range::create(context))
            );
        else
            return context->equality_expr().front()->accept(this);
    }

    std::any AstBuilder::visitEquality_expr(FSharpParser::Equality_exprContext* context)
    {
        std::vector<ast_ptr<Expression>> results;
        results.reserve(context->children.size() / 2 + 1);

        std::vector<Expression::OP::EqualityType> operators;
        operators.reserve(context->children.size() / 2);

        for (const auto child : context->children)
        {
            if (const auto expression = dynamic_cast<decltype(context->relation_expr(0))>(child))
            {
                if (auto result = expression->accept(this);
                    result.has_value())
                    results.push_back(ast::any_cast<Expression>(result, context));
            }
            else if (const auto op = dynamic_cast<tree::TerminalNode*>(child))
            {
                switch (op->getSymbol()->getType())
                {
                case FSharpParser::EQUALS:
                    operators.push_back(Expression::OP::EqualityType::EQUAL);
                    break;
                case FSharpParser::NOT_EQ:
                    operators.push_back(Expression::OP::EqualityType::NOT_EQUAL);
                    break;
                default:
                    throw std::runtime_error("Invalid Equality Expr");
                }
            }
        }

        if (context->relation_expr().size() > 1)
            return make_ast<Expression>(
                Expression::OP(
                    std::move(results),
                    Expression::OP::Type::EQUALITY,
                    std::move(operators),
                    Range::create(context))
            );
        else
            return context->relation_expr().front()->accept(this);
    }

    std::any AstBuilder::visitRelation_expr(FSharpParser::Relation_exprContext* context)
    {
        std::vector<ast_ptr<Expression>> results;
        results.reserve(context->children.size() / 2 + 1);

        std::vector<Expression::OP::RelationType> operators;
        operators.reserve(context->children.size() / 2);

        for (const auto child : context->children)
        {
            if (const auto expression = dynamic_cast<decltype(context->additive_expr(0))>(child))
            {
                if (auto result = expression->accept(this);
                    result.has_value())
                    results.push_back(ast::any_cast<Expression>(result, context));
            }
            else if (const auto op = dynamic_cast<tree::TerminalNode*>(child))
            {
                switch (op->getSymbol()->getType())
                {
                case FSharpParser::LESS_THAN:
                    operators.push_back(Expression::OP::RelationType::LESS);
                    break;
                case FSharpParser::GREATER_THAN:
                    operators.push_back(Expression::OP::RelationType::GREATER);
                    break;
                case FSharpParser::LT_EQ:
                    operators.push_back(Expression::OP::RelationType::LESS_EQUAL);
                    break;
                case FSharpParser::GT_EQ:
                    operators.push_back(Expression::OP::RelationType::GREATER_EQUAL);
                    break;
                default:
                    throw std::runtime_error("Invalid Relation Expr");
                }
            }
        }

        if (context->additive_expr().size() > 1)
            return make_ast<Expression>(
                Expression::OP(
                    std::move(results),
                    Expression::OP::Type::RELATION,
                    std::move(operators),
                    Range::create(context))
            );
        else
            return context->additive_expr().front()->accept(this);
    }

    std::any AstBuilder::visitAdditive_expr(FSharpParser::Additive_exprContext* context)
    {
        std::vector<ast_ptr<Expression>> results;
        results.reserve(context->children.size() / 2 + 1);

        std::vector<Expression::OP::ArithmeticType> operators;
        operators.reserve(context->children.size() / 2);

        for (const auto child : context->children)
        {
            if (const auto expression = dynamic_cast<decltype(context->multiplicative_expr(0))>(child))
            {
                if (auto result = expression->accept(this);
                    result.has_value())
                    results.push_back(ast::any_cast<Expression>(result, context));
            }
            else if (const auto op = dynamic_cast<tree::TerminalNode*>(child))
            {
                switch (op->getSymbol()->getType())
                {
                case FSharpParser::PLUS:
                    operators.push_back(Expression::OP::ArithmeticType::ADD);
                    break;
                case FSharpParser::MINUS:
                    operators.push_back(Expression::OP::ArithmeticType::SUBTRACT);
                    break;
                default:
                    throw std::runtime_error("Invalid Additive Expr");
                }
            }
        }

        if (context->multiplicative_expr().size() > 1)
            return make_ast<Expression>(
                Expression::OP(
                    std::move(results),
                    Expression::OP::Type::ARITHMETIC,
                    std::move(operators),
                    Range::create(context))
            );
        else
            return context->multiplicative_expr().front()->accept(this);
    }

    std::any AstBuilder::visitMultiplicative_expr(FSharpParser::Multiplicative_exprContext* context)
    {
        std::vector<ast_ptr<Expression>> results;
        results.reserve(context->children.size() / 2 + 1);

        std::vector<Expression::OP::ArithmeticType> operators;
        operators.reserve(context->children.size() / 2);

        for (const auto child : context->children)
        {
            if (const auto expression = dynamic_cast<decltype(context->dot_get_expr(0))>(child))
            {
                if (auto result = expression->accept(this);
                    result.has_value())
                    results.push_back(ast::any_cast<Expression>(result, context));
            }
            else if (const auto op = dynamic_cast<tree::TerminalNode*>(child))
            {
                switch (op->getSymbol()->getType())
                {
                case FSharpParser::STAR:
                    operators.push_back(Expression::OP::ArithmeticType::MULTIPLY);
                    break;
                case FSharpParser::DIV:
                    operators.push_back(Expression::OP::ArithmeticType::DIVIDE);
                    break;
                case FSharpParser::MOD:
                    operators.push_back(Expression::OP::ArithmeticType::MODULO);
                    break;
                default:
                    throw std::runtime_error("Invalid Multiplicative Expr");
                }
            }
        }

        if (context->dot_get_expr().size() > 1)
            return make_ast<Expression>(
                Expression::OP(
                    std::move(results),
                    Expression::OP::Type::ARITHMETIC,
                    std::move(operators),
                    Range::create(context))
            );
        else
            return context->dot_get_expr().front()->accept(this);
    }

    std::any AstBuilder::visitDot_get_expr(FSharpParser::Dot_get_exprContext* context)
    {
        if (context->long_ident())
        {
            if (auto result = context->dot_index_get_expr()->accept(this); result.has_value())
                return make_ast<Expression>(
                    Expression::DotGet(
                        ast::any_cast<Expression>(result, context),
                        ast::any_cast<LongIdent>(context->long_ident()->accept(this), context),
                        Range::create(context)
                    )
                );
        }
        return context->dot_index_get_expr()->accept(this);
    }

    std::any AstBuilder::visitDot_index_get_expr(FSharpParser::Dot_index_get_exprContext* context)
    {
        if (context->typed_expr().size() > 1)
        {
            auto base_result = context->typed_expr().front()->accept(this);
            auto index_result = context->typed_expr().back()->accept(this);
            return make_ast<Expression>(
                Expression::DotIndexedGet(
                    ast::any_cast<Expression>(base_result, context),
                    ast::any_cast<Expression>(index_result, context),
                    Range::create(context)
                )
            );
        }
        else
            return context->typed_expr().front()->accept(this);
    }

    std::any AstBuilder::visitTyped_expr(FSharpParser::Typed_exprContext* context)
    {
        auto result = context->unary_expression()->accept(this);

        if (context->type())
        {
            auto type_result = context->type()->accept(this);
            return make_ast<Expression>(
                Expression::Typed(
                    ast::any_cast<Expression>(result, context),
                    ast::any_cast<Type>(type_result, context),
                    Range::create(context))
            );
        }
        return ast::any_cast<Expression>(result, context);
    }

    std::any AstBuilder::visitUnary_expression(FSharpParser::Unary_expressionContext* context)
    {
        if (context->unary_expression())
        {
            Expression::Unary::Type type;
            if (context->MINUS())
                type = Expression::Unary::Type::MINUS;
            else if (context->PLUS())
                type = Expression::Unary::Type::PLUS;
            else
                type = Expression::Unary::Type::NOT;

            auto result = context->unary_expression()->accept(this);
            return make_ast<Expression>(
                Expression::Unary(
                    ast::any_cast<Expression>(result, context),
                    type,
                    Range::create(context)
                )
            );
        }
        return context->atomic_expr()->accept(this);
    }

    std::any AstBuilder::visitAtomic_expr(FSharpParser::Atomic_exprContext* context)
    {
        return context->children[0]->accept(this);
    }

    std::any AstBuilder::visitParen_expr(FSharpParser::Paren_exprContext* context)
    {
        auto result = context->expression()->accept(this);
        return make_ast<Expression>(
            Expression::Paren(
                ast::any_cast<Expression>(result, context),
                Range::create(context))
        );
    }

    std::any AstBuilder::visitConstant_expr(FSharpParser::Constant_exprContext* context)
    {
        return make_ast<Expression>(
            Expression::Constant(
                ast::any_cast<Constant>(context->constant()->accept(this), context))
        );
    }

    std::any AstBuilder::visitIdent_expr(FSharpParser::Ident_exprContext* context)
    {
        return make_ast<Expression>(
            Expression::Ident(
                ast::any_cast<Ident>(context->ident()->accept(this), context))
        );
    }

    std::any AstBuilder::visitLong_ident_expr(FSharpParser::Long_ident_exprContext* context)
    {
        return make_ast<Expression>(
            Expression::LongIdent(
                ast::any_cast<LongIdent>(context->long_ident()->accept(this), context))
        );
    }

    std::any AstBuilder::visitNull_expr(FSharpParser::Null_exprContext* context)
    {
        return make_ast<Expression>(Expression::Null(Range::create(context)));
    }

    std::any AstBuilder::visitRecord_expr(FSharpParser::Record_exprContext* context)
    {
        std::vector<Expression::Record::Field> fields;
        for (const auto& record_expr_field : context->record_expr_field())
        {
            fields.push_back(std::any_cast<Expression::Record::Field>(record_expr_field->accept(this)));
        }

        return make_ast<Expression>(Expression::Record(
                std::move(fields),
                Range::create(context))
        );
    }

    std::any AstBuilder::visitRecord_expr_field(FSharpParser::Record_expr_fieldContext* context)
    {
        return Expression::Record::Field(
            ast::any_cast<Ident>(context->ident()->accept(this), context),
            ast::any_cast<Expression>(context->expression()->accept(this), context)
        );
    }

    std::any AstBuilder::visitArray_expr(FSharpParser::Array_exprContext* context)
    {
        std::vector<ast_ptr<Expression>> expressions;
        for (const auto expression : context->expression())
        {
            expressions.emplace_back(ast::any_cast<Expression>(expression->accept(this), context));
        }
        return make_ast<Expression>(Expression::Array(
                std::move(expressions), Range::create(context))
        );
    }

    std::any AstBuilder::visitList_expr(FSharpParser::List_exprContext* context)
    {
        std::vector<ast_ptr<Expression>> expressions;
        for (const auto expression : context->expression())
        {
            expressions.emplace_back(ast::any_cast<Expression>(expression->accept(this), context));
        }
        return make_ast<Expression>(Expression::List(
                std::move(expressions), Range::create(context))
        );
    }

    std::any AstBuilder::visitNew_expr(FSharpParser::New_exprContext* context)
    {
        auto type = ast::any_cast<fsharpgrammar::Type>(context->type()->accept(this), context);
        std::optional<ast_ptr<Expression>> expression{};
        if (context->expression())
        {
            expression = ast::any_cast<Expression>(context->expression()->accept(this), context);
        }
        return make_ast<Expression>(Expression::New(
                std::move(type),
                std::move(expression),
                Range::create(context))
        );
    }

    std::any AstBuilder::visitIf_then_else_expr(FSharpParser::If_then_else_exprContext* context)
    {
        auto condition = ast::any_cast<Expression>(context->expression()->accept(this), context);
        auto then_body = std::any_cast<std::vector<ast_ptr<Expression>>>(context->body(0)->accept(this));
        std::optional<std::vector<ast_ptr<Expression>>> else_body{};
        if (context->ELSE())
        {
            else_body = std::any_cast<std::vector<ast_ptr<Expression>>>(context->body(1)->accept(this));
        }
        return make_ast<Expression>(
            Expression::IfThenElse(
                std::move(condition),
                std::move(then_body),
                std::move(else_body),
                Range::create(context))
        );
    }

    std::any AstBuilder::visitMatch_expr(FSharpParser::Match_exprContext* context)
    {
        auto expression = ast::any_cast<Expression>(context->expression()->accept(this), context);
        std::vector<ast_ptr<MatchClause>> match_clauses;
        for (auto match_clause_stmt : context->match_clause_stmt())
        {
            match_clauses.push_back(
                ast::any_cast<MatchClause>(match_clause_stmt->accept(this), context)
            );
        }

        return make_ast<Expression>(
            Expression::Match(
                std::move(expression),
                std::move(match_clauses),
                Range::create(context))
        );
    }

    std::any AstBuilder::visitMatch_clause_stmt(FSharpParser::Match_clause_stmtContext* context)
    {
        auto pattern = ast::any_cast<Pattern>(context->pattern()->accept(this), context);
        std::optional<ast_ptr<Expression>> expression{};
        if (context->expression())
        {
            expression = ast::any_cast<Expression>(context->expression()->accept(this), context);
        }
        auto body_expressions = std::any_cast<std::vector<ast_ptr<Expression>>>(context->body()->accept(this));

        return make_ast<MatchClause>(
            std::move(pattern),
            std::move(expression),
            std::move(body_expressions),
            Range::create(context)
        );
    }

    std::any AstBuilder::visitPipe_right_expr(FSharpParser::Pipe_right_exprContext* context)
    {
        auto previous_expression = std::shared_ptr<Expression>(nullptr);
        return make_ast<Expression>(Expression::PipeRight(
                std::move(previous_expression),
                std::any_cast<std::vector<ast_ptr<Expression>>>(context->body()->accept(this)),
                Range::create(context))
        );
    }

    std::any AstBuilder::visitAssignment_expr(FSharpParser::Assignment_exprContext* context)
    {
        return context->children[0]->accept(this);
    }

    std::any AstBuilder::visitLet_expr(FSharpParser::Let_exprContext* context)
    {
        auto binding = ast::any_cast<Pattern>(context->binding()->accept(this), context);
        auto expressions = std::any_cast<std::vector<ast_ptr<Expression>>>(context->body()->accept(this));
        return make_ast<Expression>(
            Expression::Let(
                context->binding()->MUTABLE() != nullptr,
                context->binding()->REC() != nullptr,
                std::move(binding),
                std::move(expressions),
                Range::create(context))
        );
    }

    std::any AstBuilder::visitLong_ident_set_expr(FSharpParser::Long_ident_set_exprContext* context)
    {
        auto long_ident = ast::any_cast<LongIdent>(context->long_ident()->accept(this), context);
        auto expression = ast::any_cast<Expression>(context->expression()->accept(this), context);
        return make_ast<Expression>(Expression::LongIdentSet(
                std::move(long_ident),
                std::move(expression),
                Range::create(context))
        );
    }

    std::any AstBuilder::visitSet_expr(FSharpParser::Set_exprContext* context)
    {
        auto target_expression = ast::any_cast<Expression>(context->atomic_expr()->accept(this), context);
        auto expression = ast::any_cast<Expression>(context->expression()->accept(this), context);
        return make_ast<Expression>(Expression::Set(
                std::move(target_expression),
                std::move(expression),
                Range::create(context))
        );
    }

    std::any AstBuilder::visitDot_set_expr(FSharpParser::Dot_set_exprContext* context)
    {
        auto target_expression = ast::any_cast<Expression>(context->atomic_expr()->accept(this), context);
        auto long_ident = ast::any_cast<LongIdent>(context->long_ident()->accept(this), context);
        auto expression = ast::any_cast<Expression>(context->expression()->accept(this), context);
        return make_ast<Expression>(Expression::DotSet(
                std::move(target_expression),
                std::move(long_ident),
                std::move(expression),
                Range::create(context))
        );
    }

    std::any AstBuilder::visitDot_index_set_expr(FSharpParser::Dot_index_set_exprContext* context)
    {
        auto pre_bracket_expressions = ast::any_cast<Expression>(context->atomic_expr()->accept(this), context);
        std::vector<ast_ptr<Expression>> bracket_expressions;
        for (auto expression : context->expression())
        {
            if (expression == context->expression().back())
                continue;
            bracket_expressions.push_back(ast::any_cast<Expression>(expression->accept(this), context));
        }
        auto expression = ast::any_cast<Expression>(context->expression().back()->accept(this), context);

        return make_ast<Expression>(Expression::DotIndexSet(
                std::move(pre_bracket_expressions),
                std::move(bracket_expressions),
                std::move(expression),
                Range::create(context))
        );
    }

    std::any AstBuilder::visitPattern(FSharpParser::PatternContext* context)
    {
        return make_ast<Pattern>(
            Pattern::Type::AndPattern,
            std::vector<ast_ptr<Pattern>>{},
            Range::create(context));
    }

    std::any AstBuilder::visitTuple_pat(FSharpParser::Tuple_patContext* context)
    {
        std::vector<ast_ptr<Pattern>> patterns;
        if (context->and_pat().size() > 1)
        {
            for (auto pat : context->and_pat())
            {
                auto result = pat->accept(this);
                patterns.push_back(ast::any_cast<Pattern>(result, context));
            }
        }

        return make_ast<Pattern>(
            Pattern::Type::TuplePattern,
            std::move(patterns),
            Range::create(context)
        );
    }

    std::any AstBuilder::visitType(FSharpParser::TypeContext* context)
    {
        return make_ast<Type>(Range::create(context));
    }

    std::any AstBuilder::visitConstant(FSharpParser::ConstantContext* context)
    {
        if (context->INTEGER())
            return make_ast<Constant>(std::stoi(context->INTEGER()->getText()), Range::create(context));
        if (context->FLOAT_NUMBER())
            return make_ast<Constant>(std::stof(context->FLOAT_NUMBER()->getText()), Range::create(context));
        if (context->STRING())
            return make_ast<Constant>(context->STRING()->getText(), Range::create(context));
        if (context->CHARACTER())
            return make_ast<Constant>(context->CHARACTER()->getText()[0], Range::create(context));
        if (context->BOOL())
            return make_ast<Constant>(context->BOOL()->getText() == "true", Range::create(context));

        return make_ast<Constant>(std::optional<Constant::Type>{}, Range::create(context));
    }

    std::any AstBuilder::visitIdent(FSharpParser::IdentContext* context)
    {
        return make_ast<Ident>(context->IDENT()->getText(), Range::create(context));
    }

    std::any AstBuilder::visitLong_ident(FSharpParser::Long_identContext* context)
    {
        std::vector<ast_ptr<Ident>> idents;
        for (auto ident : context->ident())
        {
            idents.push_back(ast::any_cast<Ident>(ident->accept(this), context));
        }
        return make_ast<LongIdent>(std::move(idents), Range::create(context));
    }
} // fsharpgrammar
