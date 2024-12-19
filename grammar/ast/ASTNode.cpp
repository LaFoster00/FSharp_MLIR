//
// Created by lasse on 12/18/24.
//

#include "ASTNode.h"

namespace fsharpgrammar
{
    ModuleOrNamespace::ModuleOrNamespace(
        const Type type,
        std::optional<std::string> name,
        std::vector<ast_ptr<ModuleDeclaration>>&& module_decls,
        Range&& range)
        :
        type(type),
        name(std::move(name)),
        module_decls(std::move(module_decls)),
        range(range)
    {
    }

    ModuleDeclaration::ModuleDeclaration(
        std::vector<ast_ptr<Expression>>&& expressions,
        Range&& range)
        :
    expressions(std::move(expressions)),
    range(range)
    {
    }

    NestedModuleDeclaration::NestedModuleDeclaration(
        const std::string_view name,
        std::vector<ast_ptr<ModuleDeclaration>>&& module_decls,
        Range&& range)
        :
        name(name),
        module_decls(std::move(module_decls)),
        range(range)
    {
    }

    Expression::Expression(Range&& range)
        :
    range(range)
    {
    }

    Main::Main(
        std::vector<ast_ptr<ModuleOrNamespace>>&& modules_or_namespaces,
        Range&& range)
        :
        modules_or_namespaces(std::move(modules_or_namespaces)),
        range(range)
    {
    }

    std::string to_string(const ModuleOrNamespace& moduleOrNamespace)
    {
        switch (moduleOrNamespace.type)
        {
        case ModuleOrNamespace::Type::NamedModule:
            return "Module " + moduleOrNamespace.name.value();
        case ModuleOrNamespace::Type::AnonymousModule:
            return "AnonymousModule";
        case ModuleOrNamespace::Type::Namespace:
            return "Namespace " + moduleOrNamespace.name.value();
        }
    }

    std::string to_string(const ModuleDeclaration& moduleDeclaration)
    {
        return "ModuleDeclaration";
    }

    std::string to_string(const NestedModuleDeclaration& nestedModuleDeclaration)
    {
        return "NestedModuleDeclaration";
    }

    std::string to_string(const Expression& expression)
    {
        return "Expression";
    }
} // fsharpgrammar
