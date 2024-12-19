//
// Created by lasse on 12/18/24.
//

#include "AstBuilder.h"

#include "ASTNode.h"
#include "ASTHelper.h"

namespace fsharpgrammar {
    std::any AstBuilder::visitMain(FSharpParser::MainContext* ctx)
    {
        std::vector<ModuleOrNamespace> anon_modules;
        for (auto child : ctx->children)
        {
            if (auto anon_module = dynamic_cast<FSharpParser::AnonmoduleContext*>(child))
            {
                std::any anon_module_result = anon_module->accept(this);
                if (anon_module_result.has_value())
                {
                    auto result = std::any_cast<ModuleOrNamespace>(anon_module_result);
                    anon_modules.push_back(result);
                }
            }
        }
        return Main(anon_modules, Range::create(ctx));
    }

    std::any AstBuilder::visitAnonmodule(FSharpParser::AnonmoduleContext* context)
    {
        return ModuleOrNamespace(
            ModuleOrNamespace::Type::AnonymousModule,
            {},
            Range::create(context));
    }

    std::any AstBuilder::visitNamedmodule(FSharpParser::NamedmoduleContext* context)
    {
        return ModuleOrNamespace(
            ModuleOrNamespace::Type::NamedModule,
            ast::to_string(context->long_ident()),
            Range::create(context));
    }*

    std::any AstBuilder::visitNamespace(FSharpParser::NamespaceContext* ctx)
    {
        return ModuleOrNamespace(
            ModuleOrNamespace::Type::Namespace,
            ast::to_string(ctx->long_ident()),
            Range::create(ctx));
    }

    std::any AstBuilder::visitNested_module(FSharpParser::Nested_moduleContext* context)
    {
        return FSharpParserBaseVisitor::visitNested_module(context);
    }

    std::any AstBuilder::visitExpression_stmt(FSharpParser::Expression_stmtContext* context)
    {
        return FSharpParserBaseVisitor::visitExpression_stmt(context);
    }
} // fsharpgrammar