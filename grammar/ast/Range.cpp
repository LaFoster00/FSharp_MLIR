//
// Created by lasse on 12/18/24.
//

#include "Range.h"

#include <ParserRuleContext.h>
#include <Token.h>

namespace fsharpgrammar
{
    Range Range::create(const antlr4::ParserRuleContext* ctx)
    {
        return Range::create(
            static_cast<int32_t>(ctx->start->getLine()),
            static_cast<int32_t>(ctx->start->getCharPositionInLine()),
            static_cast<int32_t>(ctx->stop->getLine()),
            static_cast<int32_t>(ctx->stop->getCharPositionInLine()));
    }
}
