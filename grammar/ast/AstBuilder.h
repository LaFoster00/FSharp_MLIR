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

        // pattern
        std::any visitTuple_pat(FSharpParser::Tuple_patContext* context) override;
    };

} // fsharpgrammar

