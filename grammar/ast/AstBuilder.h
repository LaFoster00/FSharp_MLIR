//
// Created by lasse on 12/18/24.
//
#pragma once

#include "FSharpParser.h"
#include "FSharpParserBaseVisitor.h"

namespace fsharpgrammar::ast
{
    class Main;
}

namespace fsharpgrammar::ast
{
    using namespace antlr4;

    class AstBuilder final : FSharpParserBaseVisitor
    {
    public:
        std::unique_ptr<Main> BuildAst(FSharpParser::MainContext* ctx);

    protected:
        // main
        // Returns a raw pointer. Should be converted to a unique_ptr immediately.
        std::any visitMain(FSharpParser::MainContext* ctx) override;

        // module_or_namespace
        std::any visitAnonmodule(FSharpParser::AnonmoduleContext* context) override;
        std::any visitNamedmodule(FSharpParser::NamedmoduleContext* context) override;
        std::any visitNamespace(FSharpParser::NamespaceContext* ctx) override;

        // module_decl
        std::any visitEmply_lines(FSharpParser::Emply_linesContext* context) override;
        std::any visitNested_module(FSharpParser::Nested_moduleContext* context) override;
        std::any visitExpression_stmt(FSharpParser::Expression_stmtContext* context) override;
        std::any visitOpen_stmt(FSharpParser::Open_stmtContext* context) override;

        // Body
        std::any visitMultiline_body(FSharpParser::Multiline_bodyContext* context) override;
        std::any visitSingle_line_body(FSharpParser::Single_line_bodyContext* context) override;

        // expression
        std::any visitInline_sequential_stmt(FSharpParser::Inline_sequential_stmtContext* context) override;
        std::any visitSequential_stmt(FSharpParser::Sequential_stmtContext* context) override;
        std::any visitExpression(FSharpParser::ExpressionContext* ctx) override;

        // non assignment expression
        std::any visitNon_assigment_expr(FSharpParser::Non_assigment_exprContext* context) override;
        std::any visitApp_expr(FSharpParser::App_exprContext* context) override;
        std::any visitTuple_expr(FSharpParser::Tuple_exprContext* context) override;
        std::any visitOr_expr(FSharpParser::Or_exprContext* context) override;
        std::any visitAnd_expr(FSharpParser::And_exprContext* context) override;
        std::any visitEquality_expr(FSharpParser::Equality_exprContext* context) override;
        std::any visitRelation_expr(FSharpParser::Relation_exprContext* context) override;
        std::any visitAdditive_expr(FSharpParser::Additive_exprContext* context) override;
        std::any visitMultiplicative_expr(FSharpParser::Multiplicative_exprContext* context) override;
        std::any visitDot_get_expr(FSharpParser::Dot_get_exprContext* context) override;
        std::any visitDot_index_get_expr(FSharpParser::Dot_index_get_exprContext* context) override;
        std::any visitTyped_expr(FSharpParser::Typed_exprContext* context) override;
        std::any visitUnary_expression(FSharpParser::Unary_expressionContext* context) override;

        // atomic
        std::any visitAtomic_expr(FSharpParser::Atomic_exprContext* context) override;
        std::any visitParen_expr(FSharpParser::Paren_exprContext* context) override;
        std::any visitConstant_expr(FSharpParser::Constant_exprContext* context) override;
        std::any visitIdent_expr(FSharpParser::Ident_exprContext* context) override;
        std::any visitLong_ident_expr(FSharpParser::Long_ident_exprContext* context) override;
        std::any visitNull_expr(FSharpParser::Null_exprContext* context) override;
        std::any visitRecord_expr(FSharpParser::Record_exprContext* context) override;
        std::any visitRecord_expr_field(FSharpParser::Record_expr_fieldContext* context) override;
        std::any visitArray_expr(FSharpParser::Array_exprContext* context) override;
        std::any visitList_expr(FSharpParser::List_exprContext* context) override;
        std::any visitNew_expr(FSharpParser::New_exprContext* context) override;
        std::any visitIf_then_else_expr(FSharpParser::If_then_else_exprContext* context) override;
        std::any visitMatch_expr(FSharpParser::Match_exprContext* context) override;
        std::any visitMatch_clause_stmt(FSharpParser::Match_clause_stmtContext* context) override;
        std::any visitPipe_right_expr(FSharpParser::Pipe_right_exprContext* context) override;

        // assignment expression
        std::any visitAssignment_expr(FSharpParser::Assignment_exprContext* context) override;
        std::any visitLet_expr(FSharpParser::Let_exprContext* context) override;
        std::any visitBinding(FSharpParser::BindingContext* context) override;
        std::any visitLong_ident_set_expr(FSharpParser::Long_ident_set_exprContext* context) override;
        std::any visitSet_expr(FSharpParser::Set_exprContext* context) override;
        std::any visitDot_set_expr(FSharpParser::Dot_set_exprContext* context) override;
        std::any visitDot_index_set_expr(FSharpParser::Dot_index_set_exprContext* context) override;

        // type
        std::any visitType(FSharpParser::TypeContext* context) override;
        std::any visitFun_type(FSharpParser::Fun_typeContext* context) override;
        std::any visitTuple_type(FSharpParser::Tuple_typeContext* context) override;
        // append type
        std::any visitPostfix_type(FSharpParser::Postfix_typeContext* context) override;
        std::any visitParen_postfix_type(FSharpParser::Paren_postfix_typeContext* context) override;
        // type
        std::any visitArray_type(FSharpParser::Array_typeContext* context) override;
        std::any visitAtomic_type(FSharpParser::Atomic_typeContext* context) override;
        std::any visitParen_type(FSharpParser::Paren_typeContext* context) override;
        std::any visitVar_type(FSharpParser::Var_typeContext* context) override;
        std::any visitLong_ident_type(FSharpParser::Long_ident_typeContext* context) override;
        std::any visitAnon_type(FSharpParser::Anon_typeContext* context) override;
        std::any visitStatic_constant_type(FSharpParser::Static_constant_typeContext* context) override;
        std::any visitStatic_constant_null_type(FSharpParser::Static_constant_null_typeContext* context) override;

        // universally used
        std::any visitConstant(FSharpParser::ConstantContext* context) override;
        std::any visitIdent(FSharpParser::IdentContext* context) override;
        std::any visitLong_ident(FSharpParser::Long_identContext* context) override;

        // pattern
        std::any visitPattern(FSharpParser::PatternContext* context) override;
        std::any visitTuple_pat(FSharpParser::Tuple_patContext* context) override;
        std::any visitAnd_pat(FSharpParser::And_patContext* context) override;
        std::any visitOr_pat(FSharpParser::Or_patContext* context) override;
        std::any visitAs_pat(FSharpParser::As_patContext* context) override;
        std::any visitCons_pat(FSharpParser::Cons_patContext* context) override;
        std::any visitTyped_pat(FSharpParser::Typed_patContext* context) override;
        std::any visitAtomic_pat(FSharpParser::Atomic_patContext* context) override;
        std::any visitParen_pat(FSharpParser::Paren_patContext* context) override;
        std::any visitAnon_pat(FSharpParser::Anon_patContext* context) override;
        std::any visitConstant_pat(FSharpParser::Constant_patContext* context) override;
        std::any visitNamed_pat(FSharpParser::Named_patContext* context) override;
        std::any visitRecord_pat(FSharpParser::Record_patContext* context) override;
        std::any visitArray_pat(FSharpParser::Array_patContext* context) override;
        std::any visitLong_ident_pat(FSharpParser::Long_ident_patContext* context) override;
        std::any visitNull_pat(FSharpParser::Null_patContext* context) override;
    };
} // fsharpgrammar
