//
// Created by lasse on 12/18/24.
//

#include "AstBuilder.h"

#include <boost/algorithm/string.hpp>

#include "ASTNode.h"
#include "ASTHelper.h"
#include "utils/FunctionTimer.h"

namespace fsharpgrammar::ast
{
    std::unique_ptr<Main> AstBuilder::BuildAst(FSharpParser::MainContext* ctx)
    {
        utils::FunctionTimer timer("BuildAst");
        return std::unique_ptr<Main>(std::any_cast<Main*>(visitMain(ctx)));
    }

    std::any AstBuilder::visitMain(FSharpParser::MainContext* ctx)
    {
        std::vector<ast_ptr<ModuleOrNamespace>> anon_modules;
        for (auto module_or_namespace : ctx->module_or_namespace())
        {
            anon_modules.push_back(ast::any_cast<ModuleOrNamespace>(module_or_namespace->accept(this), ctx));
        }
        return new Main(std::move(anon_modules), Range::create(ctx));
    }

    std::vector<ast_ptr<ModuleDeclaration>> get_module_declarations(
        const std::vector<FSharpParser::Module_declContext*>& decls,
        antlr4::ParserRuleContext* context,
        FSharpParserVisitor* visitor)
    {
        std::vector<ast_ptr<ModuleDeclaration>> module_declarations;
        module_declarations.reserve(decls.size());
        for (const auto module_decl : decls)
        {
            if (auto result = module_decl->accept(visitor); result.has_value())
                module_declarations.push_back(ast::any_cast<ModuleDeclaration>(std::move(result), context));
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

    std::any AstBuilder::visitEmply_lines(FSharpParser::Emply_linesContext* context)
    {
        return {};
    }

    std::any AstBuilder::visitNested_module(FSharpParser::Nested_moduleContext* context)
    {
        ModuleDeclaration::NestedModule nested_module(
            ast::any_cast<LongIdent>(context->long_ident()->accept(this), context),
            get_module_declarations(context->module_decl(), context, this),
            Range::create(context));

        return make_ast<ModuleDeclaration>(
            ModuleDeclaration::NestedModule(
                ast::any_cast<LongIdent>(context->long_ident()->accept(this), context),
                get_module_declarations(context->module_decl(), context, this),
                Range::create(context))
        );
    }

    std::any AstBuilder::visitExpression_stmt(FSharpParser::Expression_stmtContext* context)
    {
        return make_ast<ModuleDeclaration>(
            ModuleDeclaration::Expression(
                ast::any_cast<Expression>(context->sequential_stmt()->accept(this), context),
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
        for (const auto sequential_stmt : context->sequential_stmt())
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

    std::any AstBuilder::visitInline_sequential_stmt(FSharpParser::Inline_sequential_stmtContext* context)
    {
        std::vector<ast_ptr<Expression>> expressions;
        for (auto expression : context->expression())
        {
            expressions.emplace_back(ast::any_cast<Expression>(expression->accept(this), context));
        }
        if (expressions.size() > 1)
        {
            return make_ast<Expression>(Expression::Sequential(
                    std::move(expressions),
                    true,
                    Range::create(context))
            );
        }
        return expressions.front();
    }

    std::any AstBuilder::visitSequential_stmt(FSharpParser::Sequential_stmtContext* context)
    {
        std::vector<ast_ptr<Expression>> expressions;
        for (const auto expr : context->expression())
        {
            expressions.push_back(ast::any_cast<Expression>(expr->accept(this), context));
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
        for (const auto tuple_expr : context->or_expr())
        {
            expressions.push_back(ast::any_cast<Expression>(tuple_expr->accept(this), context));
        }

        if (expressions.size() > 1)
        {
            const auto is_function_call = ast::find_closest_parent<
                FSharpParser::Let_exprContext, FSharpParser::Module_declContext>(context->parent);
            return make_ast<Expression>(
                Expression::Append(
                    std::move(expressions),
                    is_function_call ? is_function_call.value() : true,
                    Range::create(context))
            );
        }

        return expressions.front();
    }

    std::any AstBuilder::visitTuple_expr(FSharpParser::Tuple_exprContext* context)
    {
        std::vector<ast_ptr<Expression>> expressions;
        for (const auto expr : context->app_expr())
        {
            expressions.push_back(ast::any_cast<Expression>(expr->accept(this), context));
        }
        if (expressions.size() > 1)
            return make_ast<Expression>(
                Expression::Tuple(
                    std::move(expressions),
                    Range::create(context))
            );

        return expressions.front();
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
                results.push_back(ast::any_cast<Expression>(and_expr->accept(this), context));
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
            return std::move(results.front());
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
                results.push_back(ast::any_cast<Expression>(expression->accept(this), context));
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
            return results.front();
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
                results.push_back(ast::any_cast<Expression>(expression->accept(this), context));
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
            return results.front();
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
                results.push_back(ast::any_cast<Expression>(expression->accept(this), context));
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
            return std::move(results.front());
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
                results.push_back(ast::any_cast<Expression>(expression->accept(this), context));
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
            return std::move(results.front());
    }

    std::any AstBuilder::visitMultiplicative_expr(FSharpParser::Multiplicative_exprContext* context)
    {
        std::vector<ast_ptr<Expression>> results;
        std::vector<Expression::OP::ArithmeticType> operators;

        for (const auto child : context->children)
        {
            if (const auto expression = dynamic_cast<decltype(context->dot_get_expr(0))>(child))
            {
                results.push_back(ast::any_cast<Expression>(expression->accept(this), context));
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
            return std::move(results.front());
    }

    std::any AstBuilder::visitDot_get_expr(FSharpParser::Dot_get_exprContext* context)
    {
        if (context->long_ident())
        {
            return make_ast<Expression>(
                Expression::DotGet(
                    ast::any_cast<Expression>(context->dot_index_get_expr()->accept(this), context),
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
            return make_ast<Expression>(
                Expression::DotIndexedGet(
                    ast::any_cast<Expression>(context->typed_expr().front()->accept(this), context),
                    ast::any_cast<Expression>(context->typed_expr().back()->accept(this), context),
                    Range::create(context)
                )
            );
        }
        else
            return context->typed_expr().front()->accept(this);
    }

    std::any AstBuilder::visitTyped_expr(FSharpParser::Typed_exprContext* context)
    {
        if (context->type())
        {
            return make_ast<Expression>(
                Expression::Typed(
                    ast::any_cast<Expression>(context->unary_expression()->accept(this), context),
                    ast::any_cast<Type>(context->type()->accept(this), context),
                    Range::create(context))
            );
        }
        return ast::any_cast<Expression>(context->unary_expression()->accept(this), context);
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

            return make_ast<Expression>(
                Expression::Unary(
                    ast::any_cast<Expression>(context->unary_expression()->accept(this), context),
                    type,
                    Range::create(context)
                )
            );
        }
        return context->atomic_expr()->accept(this);
    }

    std::any AstBuilder::visitAtomic_expr(FSharpParser::Atomic_exprContext* context)
    {
        return context->children.front()->accept(this);
    }

    std::any AstBuilder::visitParen_expr(FSharpParser::Paren_exprContext* context)
    {
        return make_ast<Expression>(
            Expression::Paren(
                ast::any_cast<Expression>(context->expression()->accept(this), context),
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
        auto type = ast::any_cast<Type>(context->type()->accept(this), context);
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
        return context->children.front()->accept(this);
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

    std::any AstBuilder::visitBinding(FSharpParser::BindingContext* context)
    {
        return context->pattern()->accept(this);
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

    std::any AstBuilder::visitType(FSharpParser::TypeContext* context)
    {
        return context->fun_type()->accept(this);
    }

    std::any AstBuilder::visitFun_type(FSharpParser::Fun_typeContext* context)
    {
        if (context->type().empty())
            return context->tuple_type()->accept(this);

        std::vector<ast_ptr<Type>> types;
        for (const auto type : context->type())
        {
            types.push_back(ast::any_cast<Type>(type->accept(this), context));
        }

        return make_ast<Type>(Type::Fun(
                ast::any_cast<Type>(context->tuple_type()->accept(this), context),
                std::move(types),
                Range::create(context))
        );
    }

    std::any AstBuilder::visitTuple_type(FSharpParser::Tuple_typeContext* context)
    {
        if (context->type().empty())
        {
            return context->append_type()->accept(this);
        }

        std::vector<ast_ptr<Type>> types;
        types.push_back(ast::any_cast<Type>(context->append_type()->accept(this), context));
        for (const auto type : context->type())
        {
            types.push_back(ast::any_cast<Type>(type->accept(this), context));
        }

        return make_ast<Type>(Type::Tuple(
                std::move(types),
                Range::create(context))
        );
    }

    std::any AstBuilder::visitPostfix_type(FSharpParser::Postfix_typeContext* context)
    {
        if (context->array_type().size() > 1)
            return make_ast<Type>(Type::Postfix(
                    ast::any_cast<Type>(context->array_type().front()->accept(this), context),
                    ast::any_cast<Type>(context->array_type().back()->accept(this), context),
                    false,
                    Range::create(context))
            );
        return context->array_type().front()->accept(this);
    }

    std::any AstBuilder::visitParen_postfix_type(FSharpParser::Paren_postfix_typeContext* context)
    {
        return make_ast<Type>(Type::Postfix(
                ast::any_cast<Type>(context->paren_type()->accept(this), context),
                ast::any_cast<Type>(context->array_type()->accept(this), context),
                true,
                Range::create(context))
        );
    }

    std::any AstBuilder::visitArray_type(FSharpParser::Array_typeContext* context)
    {
        if (context->OPEN_BRACK())
            return make_ast<Type>(Type::Array(
                    ast::any_cast<Type>(context->atomic_type()->accept(this), context),
                    Range::create(context))
            );

        return context->atomic_type()->accept(this);
    }

    std::any AstBuilder::visitAtomic_type(FSharpParser::Atomic_typeContext* context)
    {
        return context->children.front()->accept(this);
    }

    std::any AstBuilder::visitParen_type(FSharpParser::Paren_typeContext* context)
    {
        return make_ast<Type>(Type::Paren(
                ast::any_cast<Type>(context->type()->accept(this), context),
                Range::create(context))
        );
    }

    std::any AstBuilder::visitVar_type(FSharpParser::Var_typeContext* context)
    {
        return make_ast<Type>(Type::Var(
                ast::any_cast<Ident>(context->ident()->accept(this), context))
        );
    }

    std::any AstBuilder::visitLong_ident_type(FSharpParser::Long_ident_typeContext* context)
    {
        return make_ast<Type>(Type::LongIdent(
                ast::any_cast<LongIdent>(context->long_ident()->accept(this), context))
        );
    }

    std::any AstBuilder::visitAnon_type(FSharpParser::Anon_typeContext* context)
    {
        return make_ast<Type>(Type::Anon(
                Range::create(context))
        );
    }

    std::any AstBuilder::visitStatic_constant_type(FSharpParser::Static_constant_typeContext* context)
    {
        return make_ast<Type>(Type::StaticConstant(
                ast::any_cast<Constant>(context->constant()->accept(this), context))
        );
    }

    std::any AstBuilder::visitStatic_constant_null_type(FSharpParser::Static_constant_null_typeContext* context)
    {
        return make_ast<Type>(Type::StaticNull(Range::create(context)));
    }

    std::string escapeSpecialCharacters(std::string s)
    {
        boost::algorithm::replace_all(s, "\\t", "\t");
        boost::algorithm::replace_all(s, "\\n", "\n");
        boost::algorithm::replace_all(s, "\\r", "\r");
        boost::algorithm::replace_all(s, "\\b", "\b");
        boost::algorithm::replace_all(s, "\\f", "\f");
        boost::algorithm::replace_all(s, "\\v", "\v");
        boost::algorithm::replace_all(s, "\\a", "\a");
        boost::algorithm::replace_all(s, "\\'", "\'");
        boost::algorithm::replace_all(s, "\\\"", "\"");
        boost::algorithm::replace_all(s, "\\\\", "\\");
        boost::algorithm::replace_first(s, "\"", "");
        boost::algorithm::replace_last(s, "\"", "");
        return s;
    }

    std::any AstBuilder::visitConstant(FSharpParser::ConstantContext* context)
    {
        if (context->INTEGER())
            return make_ast<Constant>(std::stoi(context->INTEGER()->getText()), Range::create(context));
        if (context->FLOAT_NUMBER())
            return make_ast<Constant>(std::stod(context->FLOAT_NUMBER()->getText()), Range::create(context));
        if (context->STRING())
            return make_ast<Constant>(escapeSpecialCharacters(context->STRING()->getText()), Range::create(context));
        if (context->CHARACTER())
            // Move past ' in the beginning of the parsed character text e.g 'a'[1] -> a
            return make_ast<Constant>(context->CHARACTER()->getText()[1], Range::create(context));
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

    std::any AstBuilder::visitPattern(FSharpParser::PatternContext* context)
    {
        return context->tuple_pat()->accept(this);
    }

    std::any AstBuilder::visitTuple_pat(FSharpParser::Tuple_patContext* context)
    {
        std::vector<ast_ptr<Pattern>> patterns;
        for (const auto pat : context->and_pat())
            patterns.push_back(ast::any_cast<Pattern>(pat->accept(this), context));
        if (patterns.size() > 1)
            return make_ast<Pattern>(
                Pattern::Tuple(
                    std::move(patterns),
                    Range::create(context))
            );

        return patterns.front();
    }

    std::any AstBuilder::visitAnd_pat(FSharpParser::And_patContext* context)
    {
        std::vector<ast_ptr<Pattern>> patterns;
        for (const auto pat : context->or_pat())
            patterns.push_back(ast::any_cast<Pattern>(pat->accept(this), context));
        if (patterns.size() > 1)
            return make_ast<Pattern>(
                Pattern::And(
                    std::move(patterns),
                    Range::create(context))
            );

        return patterns.front();
    }

    std::any AstBuilder::visitOr_pat(FSharpParser::Or_patContext* context)
    {
        std::vector<ast_ptr<Pattern>> patterns;
        for (const auto pat : context->as_pat())
            patterns.push_back(ast::any_cast<Pattern>(pat->accept(this), context));
        if (patterns.size() > 1)
            return make_ast<Pattern>(
                Pattern::And(
                    std::move(patterns),
                    Range::create(context))
            );

        return patterns.front();
    }

    std::any AstBuilder::visitAs_pat(FSharpParser::As_patContext* context)
    {
        std::vector<ast_ptr<Pattern>> patterns;
        for (auto pattern : context->cons_pat())
            patterns.emplace_back(ast::any_cast<Pattern>(pattern->accept(this), context));

        if (patterns.size() > 1)
            return make_ast<Pattern>(
                Pattern::As(
                    std::move(patterns.front()),
                    std::move(patterns.back()),
                    Range::create(context))
            );
        return patterns.front();
    }

    std::any AstBuilder::visitCons_pat(FSharpParser::Cons_patContext* context)
    {
        std::vector<ast_ptr<Pattern>> patterns;
        for (auto pattern : context->typed_pat())
            patterns.emplace_back(ast::any_cast<Pattern>(pattern->accept(this), context));

        if (patterns.size() > 1)
            return make_ast<Pattern>(
                Pattern::Cons(
                    std::move(patterns.front()),
                    std::move(patterns.back()),
                    Range::create(context))
            );
        return patterns.front();
    }

    std::any AstBuilder::visitTyped_pat(FSharpParser::Typed_patContext* context)
    {
        if (context->type())
            return make_ast<Pattern>(Pattern::Typed(
                    ast::any_cast<Pattern>(context->atomic_pat()->accept(this), context),
                    ast::any_cast<Type>(context->type()->accept(this), context),
                    Range::create(context))
            );

        return context->atomic_pat()->accept(this);
    }

    std::any AstBuilder::visitAtomic_pat(FSharpParser::Atomic_patContext* context)
    {
        return context->children.front()->accept(this);
    }

    std::any AstBuilder::visitParen_pat(FSharpParser::Paren_patContext* context)
    {
        return make_ast<Pattern>(Pattern::Paren(
                ast::any_cast<Pattern>(context->pattern()->accept(this), context),
                Range::create(context))
        );
    }

    std::any AstBuilder::visitAnon_pat(FSharpParser::Anon_patContext* context)
    {
        return make_ast<Pattern>(Pattern::Anon(Range::create(context)));
    }

    std::any AstBuilder::visitConstant_pat(FSharpParser::Constant_patContext* context)
    {
        return make_ast<Pattern>(Pattern::Constant(
                ast::any_cast<Constant>(context->constant()->accept(this), context))
        );
    }

    std::any AstBuilder::visitNamed_pat(FSharpParser::Named_patContext* context)
    {
        return make_ast<Pattern>(Pattern::Named(
                ast::any_cast<Ident>(context->ident()->accept(this), context))
        );
    }

    std::any AstBuilder::visitRecord_pat(FSharpParser::Record_patContext* context)
    {
        std::vector<Pattern::Record::Field> fields;
        fields.reserve(context->children.size() / 2);
        for (size_t field = 0; field < context->children.size() / 2; ++field)
        {
            fields.emplace_back(
                ast::any_cast<Ident>(context->children[field * 2 + 0]->accept(this), context),
                ast::any_cast<Pattern>(context->children[field * 2 + 1]->accept(this), context)
            );
        }
        return make_ast<Pattern>(Pattern::Record(
            std::move(fields),
            Range::create(context)
        ));
    }

    std::any AstBuilder::visitArray_pat(FSharpParser::Array_patContext* context)
    {
        std::vector<ast_ptr<Pattern>> patterns;
        for (const auto pattern : context->atomic_pat())
        {
            patterns.emplace_back(ast::any_cast<Pattern>(pattern->accept(this), context));
        }
        return make_ast<Pattern>(Pattern::Array(
                std::move(patterns),
                Range::create(context))
        );
    }

    std::any AstBuilder::visitLong_ident_pat(FSharpParser::Long_ident_patContext* context)
    {
        std::vector<ast_ptr<Pattern>> patterns;
        for (const auto pattern : context->atomic_pat())
            patterns.emplace_back(ast::any_cast<Pattern>(pattern->accept(this), context));

        return make_ast<Pattern>(Pattern::LongIdent(
                ast::any_cast<LongIdent>(context->long_ident()->accept(this), context),
                std::move(patterns),
                Range::create(context))
        );
    }

    std::any AstBuilder::visitNull_pat(FSharpParser::Null_patContext* context)
    {
        return make_ast<Pattern>(Pattern::Null(Range::create(context)));
    }
} // fsharpgrammar
