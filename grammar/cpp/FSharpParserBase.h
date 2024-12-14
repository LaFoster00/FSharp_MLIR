//
// Created by lasse on 12/14/24.
//
#pragma once

#include "antlr4-runtime.h"

class FSharpParserBase : public antlr4::Parser {
public:
    FSharpParserBase(antlr4::TokenStream *input) : Parser(input) { }
    bool CannotBePlusMinus();
    bool CannotBeDotLpEq();

};