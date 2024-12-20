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
        moduleDecls(std::move(module_decls)),
        range(range)
    {
    }

    ModuleDeclaration::NestedModule::NestedModule(
        const std::string& name,
        std::vector<ast_ptr<ModuleDeclaration>>&& module_decls,
        Range&& range):
        name(name),
        moduleDecls(std::move(module_decls)),
        range(range)
    {
    }

    ModuleDeclaration::Expression::Expression(
        ast_ptr<fsharpgrammar::Expression>&& expression,
        Range&& range)
        :
        expression(std::move(expression)),
        range(range)
    {

    }

    ModuleDeclaration::Open::Open(const std::string& module_name, Range&& range)
        :
        moduleName(module_name),
        range(range)
    {
    }

    ModuleDeclaration::ModuleDeclaration(ModuleDeclarationType &&module_decl)
        :
        declaration(std::move(module_decl))
    {
    }

    Expression::Expression(ExpressionType&& expression)
        :
    expression(std::move(expression))
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

    std::string to_string(const ModuleDeclaration::NestedModule& nestedModuleDeclaration)
    {
        return "NestedModuleDeclaration";
    }

    std::string to_string(const ModuleDeclaration& moduleDeclaration)
    {
        return "ModuleDeclaration";
    }
} // fsharpgrammar
