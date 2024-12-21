//
// Created by lasse on 12/18/24.
//

#include "ASTNode.h"

#include <sstream>
#include <utility>

namespace fsharpgrammar
{
    Main::Main(
        std::vector<ast_ptr<ModuleOrNamespace>>&& modules_or_namespaces,
        Range&& range)
        :
        modules_or_namespaces(std::move(modules_or_namespaces)),
        range(range)
    {
    }

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
        std::string name,
        std::vector<ast_ptr<ModuleDeclaration>>&& module_decls,
        Range&& range):
        name(std::move(name)),
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

    ModuleDeclaration::Open::Open(std::string module_name, Range&& range)
        :
        moduleName(std::move(module_name)),
        range(range)
    {
    }

    ModuleDeclaration::ModuleDeclaration(ModuleDeclarationType&& module_decl)
        :
        declaration(std::move(module_decl))
    {
    }

    Expression::Expression(ExpressionType&& expression)
        :
        expression(std::move(expression))
    {
    }

    Range Expression::get_range() const
    {
        return INodeAlternative::get_range(expression);
    }

    std::string to_string(const Main& main)
    {
        std::vector<std::string> child_strings;
        child_strings.reserve(main.modules_or_namespaces.size());
        for (const auto& modules_or_namespace : main.modules_or_namespaces)
        {
            child_strings.push_back(utils::to_string(*modules_or_namespace));
        }
        std::stringstream ss;
        ss << "[Main\n";
        for (const auto& child_string : child_strings)
        {
            ss << utils::indent_string(child_string, 1, false);
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
            ss << fmt::format("Module {} {}\n", moduleOrNamespace.name.value(),
                              utils::to_string(moduleOrNamespace.range));
            break;
        case ModuleOrNamespace::Type::AnonymousModule:
            ss << fmt::format("AnonymousModule {}\n", utils::to_string(moduleOrNamespace.range));
            break;
        case ModuleOrNamespace::Type::Namespace:
            ss << fmt::format("Namespace {} {}\n", moduleOrNamespace.name.value(),
                              utils::to_string(moduleOrNamespace.range));
            break;
        }

        for (const auto& module_decl : moduleOrNamespace.moduleDecls)
        {
            ss << utils::indent_string(utils::to_string(*module_decl), 1, false);
        }
        return ss.str();
    }

    std::string to_string(const ModuleDeclaration::NestedModule& nestedModuleDeclaration)
    {
        std::stringstream ss;
        ss << fmt::format("[Nested Module {} {}\n", nestedModuleDeclaration.name,
                          utils::to_string(nestedModuleDeclaration.range));
        for (const auto& module_decl : nestedModuleDeclaration.moduleDecls)
        {
            ss << utils::indent_string(utils::to_string(*module_decl));
        }
        return ss.str();
    }

    std::string to_string(const ModuleDeclaration& moduleDeclaration)
    {
        std::stringstream ss;
        ss << fmt::format("Module Declaration {}\n", utils::to_string(moduleDeclaration.get_range()));
        ss << utils::indent_string(utils::to_string(moduleDeclaration.declaration), 1, true, false);
        return ss.str();
    }

    std::string to_string(const Constant& constant)
    {
        std::stringstream ss;
        ss << fmt::format("Constant {}\n", utils::to_string(constant.range));

        if (constant.value.has_value())
        {
            auto value = std::visit(utils::overloaded{
                           [](const int32_t i) { return std::to_string(i); },
                           [](const float_t f) { return std::to_string(f); },
                           [](const std::string& s) { return s; },
                           [](const char8_t c) { return std::to_string(c); },
                           [](const bool b) { return std::to_string(b); },
                       }, constant.value.value());
            ss << utils::indent_string(value + '\n');
        }
        else
        {
            ss << utils::indent_string("");
        }
        return ss.str();
    }

    std::string to_string(const Ident& ident)
    {
        std::stringstream ss;
        ss << fmt::format("Ident {}\n{}",
            utils::to_string(ident.range.start()),
            utils::indent_string(ident.ident, 1, false));
        return ss.str();
    }

    std::string to_string(const LongIdent& longIdent)
    {
        std::stringstream ss;
        ss << fmt::format("LongIdent {}\n", utils::to_string(longIdent.range));
        std::string fullIdent = "\t";
        for (size_t i = 0; i < longIdent.idents.size(); ++i)
        {
            fullIdent += longIdent.idents[i]->ident;
            if (i < longIdent.idents.size() - 1)
                fullIdent += '.';
        }
        ss << fullIdent;
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

    std::string to_string(const Expression::Append& append)
    {
        std::stringstream ss;
        ss << fmt::format("Append {}\n", utils::to_string(append.range));
        for (const auto& expression : append.expressions)
        {
            ss << utils::indent_string(utils::to_string(*expression));
        }

        return ss.str();
    }

    std::string to_string(const Expression::Tuple& tuple)
    {
        std::stringstream ss;
        ss << fmt::format("Tuple {}\n", utils::to_string(tuple.range));
        for (size_t i = 0; i < tuple.expressions.size(); ++i)
        {
            ss << utils::indent_string(utils::to_string(*tuple.expressions[i]));
            if (i < tuple.expressions.size() - 1)
                ss << ",\n";
        }
        return ss.str();
    }

    std::string to_string(const Expression::OP& op)
    {
        std::string title;
        std::vector<std::string> operators;
        switch (op.type)
        {
        case Expression::OP::Type::LOGICAL:
            title = "Logical";
            for (const auto l_op : std::get<std::vector<Expression::OP::LogicalType>>(op.ops))
            {
                switch (l_op)
                {
                case Expression::OP::LogicalType::AND:
                    operators.push_back("&");
                    break;
                case Expression::OP::LogicalType::OR:
                    operators.push_back("|");
                    break;
                }
            }
            break;
        case Expression::OP::Type::EQUALITY:
            title = "Equality";
            for (const auto e_op : std::get<std::vector<Expression::OP::EqualityType>>(op.ops))
            {
                switch (e_op)
                {
                case Expression::OP::EqualityType::EQUAL:
                    operators.push_back("=");
                    break;
                case Expression::OP::EqualityType::NOT_EQUAL:
                    operators.push_back("!=");
                    break;
                }
            }
            break;
        case Expression::OP::Type::RELATION:
            title = "Relation";
            for (const auto r_op : std::get<std::vector<Expression::OP::RelationType>>(op.ops))
            {
                switch (r_op)
                {
                case Expression::OP::RelationType::LESS:
                    operators.push_back("<");
                    break;
                case Expression::OP::RelationType::GREATER:
                    operators.push_back(">");
                    break;
                case Expression::OP::RelationType::LESS_EQUAL:
                    operators.push_back("<=");
                    break;
                case Expression::OP::RelationType::GREATER_EQUAL:
                    operators.push_back(">=");
                    break;
                }
            }
            break;
        case Expression::OP::Type::ARITHMETIC:
            title = "Arithmetic";
            for (const auto a_op : std::get<std::vector<Expression::OP::ArithmeticType>>(op.ops))
            {
                switch (a_op)
                {
                case Expression::OP::ArithmeticType::ADD:
                    operators.push_back("+");
                    break;
                case Expression::OP::ArithmeticType::SUBTRACT:
                    operators.push_back("-");
                    break;
                case Expression::OP::ArithmeticType::MULTIPLY:
                    operators.push_back("*");
                    break;
                case Expression::OP::ArithmeticType::DIVIDE:
                    operators.push_back("/");
                    break;
                case Expression::OP::ArithmeticType::MODULO:
                    operators.push_back("%");
                    break;
                }
            }
        }

        std::stringstream ss;

        ss << fmt::format("{} {}\n", title, utils::to_string(op.range));

        for (size_t i = 0; i < op.expressions.size(); ++i)
        {
            ss << utils::indent_string(
                utils::to_string(*op.expressions[i].get()),
                2);
            if (i < op.expressions.size() - 1)
            {
                ss << '\t' << operators[i] << '\n';
            }
        }

        return ss.str();
    }

    std::string to_string(const Expression::DotGet& dot_get)
    {
        std::stringstream ss;
        ss << ".Get\n";
        ss << utils::indent_string(utils::to_string(*dot_get.expression));
        return ss.str();
    }

    std::string to_string(const Expression::DotIndexedGet& dot_get)
    {
        std::stringstream ss;
        ss << ".[Get]\n";
        ss << utils::indent_string(utils::to_string(*dot_get.base_expression), 2);
        ss << fmt::format(
            "\t[\n{}\t]\n",
            utils::indent_string(utils::to_string(*dot_get.index_expression), 2)
        );
        return ss.str();
    }

    std::string to_string(const Expression::Typed& typed)
    {
        std::stringstream ss;
        ss << "Typed\n";
        ss << utils::indent_string(utils::to_string(*typed.expression));
        ss << ':';
        ss << utils::indent_string(utils::to_string(*typed.type));
        return ss.str();
    }

    std::string to_string(const Expression::Unary& unary)
    {
        std::stringstream ss;
        ss << "Unary\n";
        switch (unary.type)
        {
        case Expression::Unary::Type::PLUS:
            ss << "-";
            break;
        case Expression::Unary::Type::MINUS:
            ss << "+";
            break;
        case Expression::Unary::Type::NOT:
            ss << "!";
            break;
        }
        ss << utils::indent_string(utils::to_string(*unary.expression));
        return ss.str();
    }

    std::string to_string(const Expression::Paren& paren)
    {
        std::stringstream ss;
        ss << fmt::format("Parenthesis {}\n", utils::to_string(paren.range));
        ss << utils::indent_string(utils::to_string(*paren.expression));
        return ss.str();
    }

    std::string to_string(const Expression::Constant& constant)
    {
        return utils::to_string(*constant.constant);
    }
} // fsharpgrammar
