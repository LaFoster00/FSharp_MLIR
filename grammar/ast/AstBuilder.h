//
// Created by lasse on 12/18/24.
//
#pragma once

#include "FSharpParser.h"
#include "FSharpParserBaseVisitor.h"

namespace fsharpgrammar {

    using namespace antlr4;

    class AstBuilder : FSharpParserBaseVisitor {
    public:
        // main
        std::any visitMain(FSharpParser::MainContext *ctx) override;

        // module_or_namespace
        std::any visitAnonmodule(FSharpParser::AnonmoduleContext* context) override;
        std::any visitNamedmodule(FSharpParser::NamedmoduleContext* context) override;
        std::any visitNamespace(FSharpParser::NamespaceContext *ctx) override;

        // module_decl
        std::any visitNested_module(FSharpParser::Nested_moduleContext* context) override;
        std::any visitExpression_stmt(FSharpParser::Expression_stmtContext* context) override;
        std::any visitOpen_stmt(FSharpParser::Open_stmtContext* context) override;

        // expression
        std::any visitSequential_stmt(FSharpParser::Sequential_stmtContext* context) override;
        std::any visitExpression(FSharpParser::ExpressionContext *ctx) override;

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
        std::any visitArray_expr(FSharpParser::Array_exprContext* context) override;
        std::any visitList_expr(FSharpParser::List_exprContext* context) override;
        std::any visitNew_expr(FSharpParser::New_exprContext* context) override;
        std::any visitIf_then_else_expr(FSharpParser::If_then_else_exprContext* context) override;
        std::any visitMatch_expr(FSharpParser::Match_exprContext* context) override;
        std::any visitPipe_right_expr(FSharpParser::Pipe_right_exprContext* context) override;

        // assignment expression
        std::any visitAssignment_expr(FSharpParser::Assignment_exprContext* context) override;
        std::any visitLet_expr(FSharpParser::Let_exprContext* context) override;

        // pattern
        std::any visitTuple_pat(FSharpParser::Tuple_patContext* context) override;

        std::any visitType(FSharpParser::TypeContext* context) override;

        // universally used
        std::any visitConstant(FSharpParser::ConstantContext* context) override;
        std::any visitIdent(FSharpParser::IdentContext* context) override;
        std::any visitLong_ident(FSharpParser::Long_identContext* context) override;
    };
} // fsharpgrammar

