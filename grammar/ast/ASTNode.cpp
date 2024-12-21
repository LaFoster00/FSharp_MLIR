//
// Created by lasse on 12/18/24.
//

#include "ASTNode.h"

#include <sstream>

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

    Pattern::Pattern(Type type, std::vector<ast_ptr<Pattern>>&& patterns, Range&& range)
    : type(type), patterns(std::move(patterns)), range(std::move(range)) {}

    Main::Main(
        std::vector<ast_ptr<ModuleOrNamespace>>&& modules_or_namespaces,
        Range&& range)
        :
        modules_or_namespaces(std::move(modules_or_namespaces)),
        range(range)
    {
    }

    std::string to_string(const Main& main)
    {
        std::vector<std::string> child_strings;
        for (auto modules_or_namespace : main.modules_or_namespaces)
        {
            child_strings.push_back(utils::to_string(*modules_or_namespace.get()));
        }
        std::stringstream ss;
        ss << "[Main\n";
        for (auto child_string : child_strings)
        {
            ss << utils::indent_string(child_string);
        }
        ss << "]\n";
        return ss.str();
    }

    std::string to_string(const ModuleOrNamespace& moduleOrNamespace)
    {
        std::stringstream ss;
        switch (moduleOrNamespace.type)
        {
        case ModuleOrNamespace::Type::NamedModule:
            ss << fmt::format("Module {} {}\n", moduleOrNamespace.name.value(), utils::to_string(moduleOrNamespace.range));
            break;
        case ModuleOrNamespace::Type::AnonymousModule:
            ss << fmt::format("AnonymousModule {}\n", utils::to_string(moduleOrNamespace.range));
            break;
        case ModuleOrNamespace::Type::Namespace:
            ss << fmt::format("Namespace {} {}\n", moduleOrNamespace.name.value(), utils::to_string(moduleOrNamespace.range));
            break;
        }

        for (auto module_decl : moduleOrNamespace.moduleDecls)
        {
            ss << utils::indent_string(utils::to_string(*module_decl.get()));
        }
        return ss.str();
    }

    std::string to_string(const ModuleDeclaration::NestedModule& nestedModuleDeclaration)
    {
        std::stringstream ss;
        ss << fmt::format("[Nested Module {} {}\n", nestedModuleDeclaration.name, utils::to_string(nestedModuleDeclaration.range));
        for (auto module_decl : nestedModuleDeclaration.moduleDecls)
        {
            ss << utils::indent_string(utils::to_string(*module_decl.get()));
        }
        return ss.str();
    }

    std::string to_string(const ModuleDeclaration& moduleDeclaration)
    {
        std::stringstream ss;
        ss << fmt::format("Module Declaration {}\n", utils::to_string(moduleDeclaration.get_range()));
        ss << utils::indent_string(utils::to_string(moduleDeclaration.declaration));
        return ss.str();
    }

    std::string to_string(const Expression::Sequential& sequential)
    {
        std::stringstream ss;
        ss << "(Sequential " + utils::to_string(sequential.range) + "\n";
        for (size_t i = 0; i < sequential.expressions.size(); ++i)
        {
            ss << utils::indent_string(utils::to_string(*sequential.expressions[i]));
            if (i < sequential.expressions.size() - 1)
                ss << ';';
        }
        ss << "\n)";
        return ss.str();
    }

    std::string to_string(const Pattern& pattern)
    {
        std::stringstream ss;
        switch (pattern.type)
        {
            case Pattern::Type::TuplePattern:
                ss << "TuplePattern(";
            for (const auto& pat : pattern.patterns)
            {
                ss << to_string(*pat) << ", ";
            }
            ss << ")";
            break;
            // handle other pattern types...
        }
        return ss.str();
    }
} // fsharpgrammar
