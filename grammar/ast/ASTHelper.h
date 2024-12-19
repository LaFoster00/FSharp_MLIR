//
// Created by lasse on 12/19/24.
//

#pragma once

#include "FSharpParser.h"

namespace fsharpgrammar::ast
{
    std::string to_string(const FSharpParser::Long_identContext* context);
    std::string to_string(const FSharpParser::IdentContext* context);
}

