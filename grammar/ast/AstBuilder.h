//
// Created by lasse on 12/18/24.
//
#pragma once

#include "antlr4-runtime.h"
#include "FSharpLexer.h"
#include "FSharpParser.h"
#include "FSharpParserBaseVisitor.h"

namespace fsharpgrammar {

    using namespace antlr4;

    class AstBuilder : FSharpParserBaseVisitor {
    public:
        std::any visitMain(FSharpParser::MainContext *ctx) override;
        std::any visitAnonmodule(FSharpParser::AnonmoduleContext* context) override;
    };

} // fsharpgrammar

