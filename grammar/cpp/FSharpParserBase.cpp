//
// Created by lasse on 12/14/24.
//
#include "FSharpParserBase.h"

using namespace antlr4;

bool FSharpParserBase::CannotBePlusMinus()
{
    return true;
}

bool FSharpParserBase::CannotBeDotLpEq()
{
    return true;
}