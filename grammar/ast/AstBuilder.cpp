//
// Created by lasse on 12/18/24.
//

#include "AstBuilder.h"

#include "ASTNode.h"

namespace fsharpgrammar {
    std::any AstBuilder::visitMain(FSharpParser::MainContext* ctx)
    {
        auto main = Main();
        auto children_content =  visitChildren(ctx);
        return main;
    }

    std::any AstBuilder::visitAnonmodule(FSharpParser::AnonmoduleContext* context)
    {
        return AnonModule();
    }
} // fsharpgrammar