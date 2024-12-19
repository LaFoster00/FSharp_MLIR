//
// Created by lasse on 12/18/24.
//

#include "AstBuilder.h"

#include "ASTNode.h"
#include "ASTHelper.h"

namespace fsharpgrammar {
    std::any AstBuilder::visitMain(FSharpParser::MainContext* ctx)
    {
        std::vector<ast_ptr<ModuleOrNamespace>> anon_modules;
        for (auto child : ctx->children)
        {
            if (auto anon_module = dynamic_cast<FSharpParser::AnonmoduleContext*>(child))
            {
                std::any anon_module_result = anon_module->accept(this);
                if (anon_module_result.has_value())
                {
                    auto result = ast::any_cast<ModuleOrNamespace>(anon_module_result, ctx);
                    //anon_modules.push_back(result);
                }
            }
        }
        return make_ast<Main>(std::move(anon_modules), Range::create(ctx));
    }

    std::any AstBuilder::visitAnonmodule(FSharpParser::AnonmoduleContext* context)
    {
        return make_ast<ModuleOrNamespace>(
            ModuleOrNamespace::Type::AnonymousModule,
            std::optional<std::string>{},
            std::vector<ast_ptr<ModuleDeclaration>>{},
            Range::create(context));
    }

    std::any AstBuilder::visitNamedmodule(FSharpParser::NamedmoduleContext* context)
    {
        return make_ast<ModuleOrNamespace>(
            ModuleOrNamespace::Type::NamedModule,
            ast::to_string(context->long_ident()),
            std::vector<ast_ptr<ModuleDeclaration>>{},
            Range::create(context));
    }

    std::any AstBuilder::visitNamespace(FSharpParser::NamespaceContext* ctx)
    {
        return make_ast<ModuleOrNamespace>(
            ModuleOrNamespace::Type::Namespace,
            ast::to_string(ctx->long_ident()),
            std::vector<ast_ptr<ModuleDeclaration>>{},
            Range::create(ctx));
    }

    std::any AstBuilder::visitNested_module(FSharpParser::Nested_moduleContext* context)
    {
        std::vector<ast_ptr<ModuleDeclaration>> module_declarations;
        for (auto module_decl : context->module_decl())
        {
            auto result = module_decl->accept(this);
            module_declarations.push_back(ast::any_cast<ModuleDeclaration>(result, context));
        }

        return make_ast<ModuleOrNamespace>(
            ModuleOrNamespace::Type::NamedModule,
            ast::to_string(context->long_ident()),
            std::vector<ast_ptr<ModuleDeclaration>>{},
            Range::create(context));
    }

    std::any AstBuilder::visitExpression_stmt(FSharpParser::Expression_stmtContext* context)
    {
        return make_ast<Expression>(Range::create(context));

    }
} // fsharpgrammar