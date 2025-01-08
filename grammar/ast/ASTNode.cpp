//
// Created by lasse on 12/18/24.
//

#include "ASTNode.h"

#include <sstream>
#include <utility>

#define RANGED_NAMED_BLOCK(Name, Object) \
    std::stringstream ss; \
    ss << fmt::format("{} {}\n", Name, utils::to_string(Object.get_range()))

namespace fsharpgrammar::ast
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
        std::optional<ast_ptr<LongIdent>> name,
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
        ast_ptr<LongIdent> name,
        std::vector<ast_ptr<ModuleDeclaration>>&& module_decls,
        Range&& range):
        name(std::move(name)),
        moduleDecls(std::move(module_decls)),
        range(range)
    {
    }

    ModuleDeclaration::Expression::Expression(
        ast_ptr<ast::Expression>&& expression,
        Range&& range)
        :
        expression(std::move(expression)),
        range(range)
    {
    }

    ModuleDeclaration::Open::Open(ast_ptr<LongIdent> module_name, Range&& range)
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

    Pattern::Pattern(PatternType&& pattern)
        :
        pattern(std::move(pattern))
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
            ss << fmt::format("Module {} {}\n", utils::to_string(*moduleOrNamespace.name.value()),
                              utils::to_string(moduleOrNamespace.range));
            break;
        case ModuleOrNamespace::Type::AnonymousModule:
            ss << fmt::format("AnonymousModule {}\n", utils::to_string(moduleOrNamespace.range));
            break;
        case ModuleOrNamespace::Type::Namespace:
            ss << fmt::format("Namespace {} {}\n", utils::to_string(*moduleOrNamespace.name.value()),
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
        ss << fmt::format("[Nested Module {} {}\n", utils::to_string(*nestedModuleDeclaration.name),
                          utils::to_string(nestedModuleDeclaration.range));
        for (const auto& module_decl : nestedModuleDeclaration.moduleDecls)
        {
            ss << utils::indent_string(utils::to_string(*module_decl));
        }
        return ss.str();
    }

    std::string to_string(const ModuleDeclaration& moduleDeclaration)
    {
        RANGED_NAMED_BLOCK("Module Declaration", moduleDeclaration);
        ss << utils::indent_string(utils::to_string(moduleDeclaration.declaration), 1, false);
        return ss.str();
    }

    std::string to_string(const Constant& constant)
    {
        RANGED_NAMED_BLOCK("Constant", constant);

        if (constant.value.has_value())
        {
            auto value = std::visit(utils::overloaded{
                                        [](const int32_t i) { return std::to_string(i); },
                                        [](const float_t f) { return std::to_string(f); },
                                        [](const std::string& s) { return s; },
                                        [](const char8_t c) { return std::to_string(c); },
                                        [](const bool b) { return std::to_string(b); },
                                    }, constant.value.value());
            ss << utils::indent_string(value + '\n', 1, false);
        }
        else
        {
            ss << utils::indent_string("");
        }
        return ss.str();
    }

    std::string to_string(const Ident& ident)
    {
        RANGED_NAMED_BLOCK("Ident Expression", ident);
        ss << utils::indent_string(ident.ident, 1, false);
        return ss.str();
    }

    std::string to_string(const LongIdent& longIdent)
    {
        RANGED_NAMED_BLOCK("LongIdent Expression", longIdent);
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
        RANGED_NAMED_BLOCK("Sequential Expression", sequential);
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
        RANGED_NAMED_BLOCK("Append Expression", append);
        for (const auto& expression : append.expressions)
        {
            ss << utils::indent_string(utils::to_string(*expression));
        }

        return ss.str();
    }

    std::string to_string(const Expression::Tuple& tuple)
    {
        RANGED_NAMED_BLOCK("Tuple Expression", tuple);
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
                    operators.emplace_back("&");
                    break;
                case Expression::OP::LogicalType::OR:
                    operators.emplace_back("|");
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
                    operators.emplace_back("=");
                    break;
                case Expression::OP::EqualityType::NOT_EQUAL:
                    operators.emplace_back("!=");
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
                    operators.emplace_back("<");
                    break;
                case Expression::OP::RelationType::GREATER:
                    operators.emplace_back(">");
                    break;
                case Expression::OP::RelationType::LESS_EQUAL:
                    operators.emplace_back("<=");
                    break;
                case Expression::OP::RelationType::GREATER_EQUAL:
                    operators.emplace_back(">=");
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
                    operators.emplace_back("+");
                    break;
                case Expression::OP::ArithmeticType::SUBTRACT:
                    operators.emplace_back("-");
                    break;
                case Expression::OP::ArithmeticType::MULTIPLY:
                    operators.emplace_back("*");
                    break;
                case Expression::OP::ArithmeticType::DIVIDE:
                    operators.emplace_back("/");
                    break;
                case Expression::OP::ArithmeticType::MODULO:
                    operators.emplace_back("%");
                    break;
                }
            }
        }


        RANGED_NAMED_BLOCK(title, op);
        for (size_t i = 0; i < op.expressions.size(); ++i)
        {
            ss << utils::indent_string(
                utils::to_string(*op.expressions[i].get()),
                1,
                true,
                true);
            if (i < op.expressions.size() - 1)
            {
                ss << '\t' << operators[i] << '\n';
            }
        }

        return ss.str();
    }

    std::string to_string(const Expression::DotGet& dot_get)
    {
        RANGED_NAMED_BLOCK("Dot Get Expression", dot_get);
        ss << utils::indent_string(utils::to_string(*dot_get.expression));
        ss << utils::indent_string(utils::to_string(*dot_get.identifier));
        return ss.str();
    }

    std::string to_string(const Expression::DotIndexedGet& dot_get)
    {
        RANGED_NAMED_BLOCK("DotIndexedGet Expression", dot_get);
        ss << utils::indent_string(utils::to_string(*dot_get.base_expression), 2);
        ss << fmt::format(
            "\t[\n{}\t]\n",
            utils::indent_string(utils::to_string(*dot_get.index_expression), 2)
        );
        return ss.str();
    }

    std::string to_string(const Expression::Typed& typed)
    {
        RANGED_NAMED_BLOCK("Typed Expression", typed);
        ss << utils::indent_string(utils::to_string(*typed.expression));
        ss << ':';
        ss << utils::indent_string(utils::to_string(*typed.type));
        return ss.str();
    }

    std::string to_string(const Expression::Unary& unary)
    {
        RANGED_NAMED_BLOCK("Unary Expression", unary);
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
        RANGED_NAMED_BLOCK("Parenthesis Expression", paren);
        ss << utils::indent_string(utils::to_string(*paren.expression));
        return ss.str();
    }

    std::string to_string(const Expression::Constant& constant)
    {
        RANGED_NAMED_BLOCK("Constant Expression", constant);
        return utils::to_string(*constant.constant);
    }

    std::string to_string(const Expression::Record& record)
    {
        RANGED_NAMED_BLOCK("Record Expression", record);

        std::stringstream args;
        for (auto& [ident, expression] : record.fields)
        {
            args << fmt::format("{}={}",
                                utils::to_string(*ident),
                                utils::to_string(*expression));
        }
        ss << utils::indent_string(
            args.str(),
            1,
            true,
            true,
            true,
            "{",
            "}");
        return ss.str();
    }

    std::string to_string(const Expression::Array& array)
    {
        RANGED_NAMED_BLOCK("Array Expression", array);

        std::stringstream args;
        for (const auto& expression : array.expressions)
        {
            args << utils::to_string(*expression) << '\n';
        }
        if (args.str().empty())
            args << ' ';

        ss << utils::indent_string(
            args.str(),
            1,
            true,
            true,
            true,
            "[",
            "]");
        return ss.str();
    }

    std::string to_string(const Expression::List& list)
    {
        RANGED_NAMED_BLOCK("List Expression", list);

        std::stringstream args;
        for (const auto& expression : list.expressions)
        {
            args << utils::to_string(*expression) << '\n';
        }
        if (args.str().empty())
            args << ' ';
        ss << utils::indent_string(
            args.str(),
            1,
            true,
            true,
            true,
            "[|",
            "|]");
        return ss.str();
    }

    std::string to_string(const Expression::New& n)
    {
        RANGED_NAMED_BLOCK("New Expression", n);
        ss << utils::indent_string(
            fmt::format("new \n{}",
                        utils::to_string(*n.type)));
        if (n.expression.has_value())
            ss << utils::indent_string(utils::to_string(*n.expression.value()));
        return ss.str();
    }

    std::string to_string(const Expression::IfThenElse& if_then_else)
    {
        RANGED_NAMED_BLOCK("Condition Expression", if_then_else);
        std::stringstream args;
        args << fmt::format("if\n{}then\n",
                            utils::indent_string(utils::to_string(*if_then_else.condition)));
        for (auto& expression : if_then_else.then)
        {
            args << utils::indent_string(utils::to_string(*expression));
        }
        if (if_then_else.else_expr.has_value())
        {
            args << "else\n";
            for (auto& expression : if_then_else.else_expr.value())
            {
                args << utils::indent_string(utils::to_string(*expression));
            }
        }
        ss << utils::indent_string(args.str());
        return ss.str();
    }

    std::string to_string(const Expression::Match& match)
    {
        RANGED_NAMED_BLOCK("Match Expression", match);
        ss << utils::indent_string(fmt::format("{}\n", utils::to_string(*match.expression)));
        for (auto& clause : match.clauses)
        {
            ss << utils::indent_string(utils::to_string(*clause), 1, false);
        }
        return ss.str();
    }

    std::string to_string(const Expression::PipeRight& right)
    {
        RANGED_NAMED_BLOCK("PipeRight Expression", right);
        for (auto& expression : right.expressions)
        {
            ss << utils::indent_string("|>" + utils::to_string(*expression));
        }

        return ss.str();
    }

    std::string to_string(const Expression::Let& let)
    {
        RANGED_NAMED_BLOCK("Let Expression", let);
        auto prefix = let.isMutable ? "mutable" : (let.isRecursive ? "recursive" : "");
        ss << utils::indent_string(
            fmt::format("{} {}", prefix, utils::to_string(*let.args))
        );
        for (auto& expression : let.expressions)
        {
            ss << utils::indent_string(utils::to_string(*expression));
        }
        return ss.str();
    }

    std::string to_string(const Expression::LongIdentSet& long_ident_set)
    {
        RANGED_NAMED_BLOCK("LongIdentSet Expression", long_ident_set);
        ss << utils::indent_string(utils::to_string(*long_ident_set.long_ident));
        ss << "<-\n";
        ss << utils::indent_string(utils::to_string(*long_ident_set.expression));
        return ss.str();
    }

    std::string to_string(const Expression::Set& set)
    {
        RANGED_NAMED_BLOCK("Set Expression", set);
        ss << utils::indent_string(utils::to_string(*set.target_expression));
        ss << "<-\n";
        ss << utils::indent_string(utils::to_string(*set.expression));
        return ss.str();
    }

    std::string to_string(const Expression::DotSet& dot_set)
    {
        RANGED_NAMED_BLOCK("DotSet Expression", dot_set);

        std::stringstream target;
        target << utils::indent_string(utils::to_string(*dot_set.target_expression));
        target << ".\n";
        target << utils::indent_string(utils::to_string(*dot_set.long_ident));

        ss << utils::indent_string("Target \n" + target.str());
        ss << "<-";
        ss << utils::indent_string(utils::to_string(*dot_set.expression));
        return ss.str();
    }

    std::string to_string(const Expression::DotIndexSet& dot_index_set)
    {
        RANGED_NAMED_BLOCK("DotIndexSet Expression", dot_index_set);

        std::stringstream target;
        target << utils::indent_string(utils::to_string(*dot_index_set.pre_bracket_expression));
        target << ".\n";

        std::stringstream bracket_expressions;
        for (auto& bracket_expression : dot_index_set.bracket_expressions)
        {
            bracket_expressions << utils::indent_string(utils::to_string(*bracket_expression));
        }
        target << utils::indent_string(bracket_expressions.str(),
                                       1, true, true, true,
                                       "[", "]");

        ss << utils::indent_string("Target \n" + target.str());
        ss << "<-";
        ss << utils::indent_string(utils::to_string(*dot_index_set.expression));
        return ss.str();
    }

    std::string to_string(const MatchClause& match_clause)
    {
        RANGED_NAMED_BLOCK("| Match Clause", match_clause);
        ss << utils::indent_string(utils::to_string(*match_clause.pattern));

        if (match_clause.when_expression.has_value())
            ss << utils::indent_string(
                fmt::format("when {}",
                            utils::to_string(*match_clause.when_expression.value())));

        for (auto& expressions : match_clause.expressions)
        {
            ss << utils::indent_string(utils::to_string(*expressions), 1);
        }

        return ss.str();
    }

    std::string to_string(const Pattern::Tuple& tuplePattern)
    {
        RANGED_NAMED_BLOCK("Tuple Pattern", tuplePattern);
        for(size_t i = 0; i < tuplePattern.patterns.size(); ++i)
        {
            ss << utils::indent_string(utils::to_string(*tuplePattern.patterns[i]));
            if(i < tuplePattern.patterns.size() - 1)
                ss << ",\n";
        }
        return ss.str();
    }

    std::string to_string(const Pattern::And& andPattern)
    {
        RANGED_NAMED_BLOCK("And Pattern", andPattern);
        for (const auto& pattern : andPattern.patterns)
        {
            ss << utils::indent_string(utils::to_string(*pattern));
        }
        return ss.str();
    }

    std::string to_string(const Pattern::Or& orPattern)
    {
        RANGED_NAMED_BLOCK("Or Pattern", orPattern);
        for (const auto& pattern : orPattern.patterns)
        {
            ss << utils::indent_string(utils::to_string(*pattern));
        }
        return ss.str();
    }

    std::string to_string(const Pattern::As& asPattern)
    {
        RANGED_NAMED_BLOCK("As Pattern", asPattern);
        ss << utils::indent_string(utils::to_string(*asPattern.left));
        ss << "as\n";
        ss << utils::indent_string(utils::to_string(*asPattern.right));
        return ss.str();
    }

    std::string to_string(const Pattern::Cons& cons)
    {
        RANGED_NAMED_BLOCK("Cons Pattern", cons);
        ss << utils::indent_string(utils::to_string(*cons.left));
        ss << "::\n";
        ss << utils::indent_string(utils::to_string(*cons.right));
        return ss.str();
    }

    std::string to_string(const Pattern::Typed& typed)
    {
        RANGED_NAMED_BLOCK("Typed Pattern", typed);
        ss << utils::indent_string(utils::to_string(*typed.pattern));
        ss << "\n";
        ss << utils::indent_string(utils::to_string(*typed.type));
        return ss.str();
    }

    std::string to_string(const Pattern::Paren& paren)
    {
        RANGED_NAMED_BLOCK("Parethesis Pattern", paren);
        ss << utils::indent_string(utils::to_string(*paren.pattern));
        return ss.str();
    }

    std::string to_string(const Pattern::Anon& anon)
    {
        RANGED_NAMED_BLOCK("Anon Pattern", anon);
        return ss.str();
    }

    std::string to_string(const Pattern::Constant& constant)
    {
        RANGED_NAMED_BLOCK("Constant Pattern", constant);
        ss << utils::indent_string(utils::to_string(*constant.constant));
        return ss.str();
    }

    std::string to_string(const Pattern::Named& named)
    {
        RANGED_NAMED_BLOCK("Named Pattern", named);
        ss << utils::indent_string(utils::to_string(*named.ident));
        return ss.str();
    }

    std::string to_string(const Pattern::LongIdent& ident)
    {
        RANGED_NAMED_BLOCK("LongIdent Pattern", ident);
        ss << utils::indent_string(utils::to_string(*ident.ident));
        for (const auto& pattern : ident.patterns)
        {
            ss << utils::indent_string(utils::to_string(*pattern));
        }
        return ss.str();
    }

    std::string to_string(const Pattern::Record& record)
    {
        RANGED_NAMED_BLOCK("Record Pattern", record);
        for (const auto& [ident, pattern] : record.fields)
        {
            ss << utils::indent_string(fmt::format("{}={}",
                                                   utils::to_string(*ident),
                                                   utils::to_string(*pattern)));
        }
        return ss.str();
    }

    std::string to_string(const Pattern::Array& array)
    {
        RANGED_NAMED_BLOCK("Array Pattern", array);
        for (const auto& pattern : array.patterns)
        {
            ss << utils::indent_string(utils::to_string(*pattern));
        }
        return ss.str();
    }

    std::string to_string(const Pattern::Null& null)
    {
        RANGED_NAMED_BLOCK("Null Pattern", null);
        return ss.str();
    }

    std::string to_string(const Type::Fun& type)
    {
        RANGED_NAMED_BLOCK("Function Type", type);
        for (const auto& fun_type : type.fun_types)
        {
            ss << utils::indent_string(fmt::format("->{}\n", utils::to_string(*fun_type)));
        }
        return ss.str();
    }

    std::string to_string(const Type::Tuple& tuple)
    {
        RANGED_NAMED_BLOCK("Tuple Type", tuple);
        for (const auto& tuple_type : tuple.types)
        {
            ss << utils::indent_string(utils::to_string(*tuple_type));
        }
        return ss.str();
    }

    std::string to_string(const Type::Postfix& postfix)
    {
        RANGED_NAMED_BLOCK(postfix.is_paren ? "Paren Type" : "", postfix);
        ss << utils::indent_string(utils::to_string(*postfix.left));
        ss << utils::indent_string(utils::to_string(*postfix.right));
        return ss.str();
    }

    std::string to_string(const Type::Array& array)
    {
        RANGED_NAMED_BLOCK("Array Type", array);
        ss << utils::indent_string(utils::to_string(*array.type));
        return ss.str();
    }

    std::string to_string(const Type::Paren& parent)
    {
        RANGED_NAMED_BLOCK("Parenthesis Type", parent);
        ss << utils::indent_string(utils::to_string(*parent.type));
        return ss.str();
    }

    std::string to_string(const Type::Var& var)
    {
        RANGED_NAMED_BLOCK("Var Type", var);
        return utils::to_string(*var.ident);
    }

    std::string to_string(const Type::LongIdent& ident)
    {
        RANGED_NAMED_BLOCK("LongIdent Type", ident);
        return utils::to_string(*ident.longIdent);
    }

    std::string to_string(const Type::Anon& anon)
    {
        RANGED_NAMED_BLOCK("Anon Type", anon);
        return ss.str();
    }

    std::string to_string(const Type::StaticConstant& constant)
    {
        RANGED_NAMED_BLOCK("StaticConstant Type", constant);
        return utils::to_string(*constant.constant);
    }

    std::string to_string(const Type::StaticNull& null)
    {
        RANGED_NAMED_BLOCK("StaticNull Type", null);
        ss << utils::indent_string("null", false);
        return ss.str();
    }
} // fsharpgrammar
