//
// Created by lasse on 12/18/24.
//

#pragma once

#include <optional>
#include <vector>

#include "Range.h"
#include "fmt/core.h"
#include <type_traits>
#include <variant>

#include "ASTNode.h"

namespace fsharpgrammar
{
    class IASTNode;

    template <typename T>
    using ast_ptr = std::shared_ptr<T>;

    template <typename T, typename... Args>
        requires std::is_base_of_v<IASTNode, T>
    auto make_ast(Args&&... args) -> decltype(std::make_shared<T>(std::forward<Args>(args)...))
    {
        return std::make_shared<T>(std::forward<Args>(args)...);
    }

    class IASTNode
    {
    public:
        friend class ASTBuilder;
        virtual ~IASTNode() = default;
        [[nodiscard]] virtual Range get_range() const = 0;
    };


    // Helper to check if all types inherit from INodeAlternative
    template <typename Base, typename... Ts>
    constexpr bool are_all_base_of = (std::is_base_of_v<Base, Ts> && ...);

    struct INodeAlternative
    {
        virtual ~INodeAlternative() = default;
        [[nodiscard]] virtual Range get_range() const = 0;

        template <typename... T>
            requires are_all_base_of<INodeAlternative, T...>
        static Range get_range(std::variant<T...> alternatives)
        {
            return std::visit([](const auto& obj)
            {
                return obj.get_range();
            }, alternatives);
        }

        template <typename... T>
            requires are_all_base_of<INodeAlternative, T...>
        friend std::string to_string(const std::variant<T...>& alternatives)
        {
            return std::visit([](const auto& obj)
            {
                return utils::to_string(obj);
            }, alternatives);
        }
    };

    struct PlaceholderNodeAlternative final : INodeAlternative
    {
        explicit PlaceholderNodeAlternative(const std::string& name) : name(name)
        {
        }

        [[nodiscard]] Range get_range() const override
        {
            return Range::create(0, 0, 0, 0);
        }

        friend std::string to_string(const PlaceholderNodeAlternative& placeholder)
        {
            return "Placeholder \n\t" + placeholder.name;
        }

        std::string name;
    };
}

template <typename T>
struct fmt::formatter<T, std::enable_if_t<std::is_base_of_v<fsharpgrammar::IASTNode, T>,
                                          char>> : fmt::formatter<std::string>
{
    auto format(const fsharpgrammar::IASTNode& node, fmt::format_context& ctx) const
    {
        return fmt::formatter<std::string>::format(utils::to_string(static_cast<const T&>(node)), ctx);
    }
};

template <typename T>
struct fmt::formatter<T, std::enable_if_t<std::is_base_of_v<fsharpgrammar::INodeAlternative, T>,
                                          char>> : fmt::formatter<std::string>
{
    auto format(const fsharpgrammar::INodeAlternative& node, fmt::format_context& ctx) const
    {
        return fmt::formatter<std::string>::format(utils::to_string(static_cast<const T&>(node)), ctx);
    }
};

namespace fsharpgrammar
{
    class ModuleOrNamespace;
    class ModuleDeclaration;
    class Expression;
    class Type;
    class Constant;
    class Ident;
    class LongIdent;
    class MatchClause;
    class Pattern;

    class Main : public IASTNode
    {
    public:
        Main(
            std::vector<ast_ptr<ModuleOrNamespace>>&& anon_modules,
            Range&& range);
        ~Main() override = default;

        [[nodiscard]] Range get_range() const override
        {
            return range;
        }

        friend std::string to_string(const Main& main);

    private:
        std::vector<ast_ptr<ModuleOrNamespace>> modules_or_namespaces;
        const Range range;
    };

    class Ident final : public IASTNode
    {
    public:
        Ident(std::string&& ident, const Range&& range)
            : ident(std::move(ident)),
              range(range)
        {
        }
        Ident(const std::string& ident, const Range& range)
            : ident(ident),
              range(range)
        {}

        friend std::string to_string(const Ident& ident);
        [[nodiscard]] Range get_range() const override { return range; }

    public:
        const std::string ident;
        const Range range;
    };

    class LongIdent final : public IASTNode
    {
    public:
        LongIdent(std::vector<ast_ptr<Ident>>&& idents, const Range&& range)
            : idents(std::move(idents)),
              range(range)
        {
        }

        friend std::string to_string(const LongIdent& ident);
        [[nodiscard]] Range get_range() const override { return range; }

    public:
        const std::vector<ast_ptr<Ident>> idents;
        const Range range;
    };

    class Constant final : public IASTNode
    {
    public:
        using Type = std::variant<
            int32_t,
            float_t,
            std::string,
            char8_t,
            bool>;

    public:
        Constant(std::optional<Type> value, Range&& range)
            : value(std::move(value)),
              range(range)
        {
        }

        friend std::string to_string(const Constant& constant);

        [[nodiscard]] Range get_range() const override { return range; }

        // If not set signals unit '()'
        const std::optional<Type> value;
        const Range range;
    };

    class ModuleOrNamespace : public IASTNode
    {
    public:
        enum class Type
        {
            NamedModule,
            AnonymousModule,
            Namespace
        };

    public:
        ModuleOrNamespace(
            Type type,
            std::optional<std::string> name,
            std::vector<ast_ptr<ModuleDeclaration>>&& module_decls,
            Range&& range);
        ~ModuleOrNamespace() override = default;

        [[nodiscard]] Range get_range() const override
        {
            return range;
        }

        friend std::string to_string(const ModuleOrNamespace& moduleOrNamespace);

        const Type type;
        const std::optional<std::string> name;
        const std::vector<ast_ptr<ModuleDeclaration>> moduleDecls;
        const Range range;
    };

    class ModuleDeclaration : public IASTNode
    {
    public:
        struct NestedModule : INodeAlternative
        {
            NestedModule(
                std::string name,
                std::vector<ast_ptr<ModuleDeclaration>>&& module_decls,
                Range&& range);

            friend std::string to_string(const NestedModule& nestedModuleDeclaration);

            [[nodiscard]] Range get_range() const override
            {
                return range;
            }

            const std::string name;
            const std::vector<ast_ptr<ModuleDeclaration>> moduleDecls;
            const Range range;
        };

        struct Expression : INodeAlternative
        {
            Expression(
                ast_ptr<fsharpgrammar::Expression>&& expression,
                Range&& range);

            friend std::string to_string(const Expression& expression)
            {
                return utils::to_string(*expression.expression.get());
            }

            [[nodiscard]] Range get_range() const override
            {
                return range;
            }

            const ast_ptr<fsharpgrammar::Expression> expression;
            const Range range;
        };

        struct Open : INodeAlternative
        {
            Open(std::string module_name, Range&& range);

            friend std::string to_string(const Open& open)
            {
                return "Open " + open.moduleName;
            }

            [[nodiscard]] Range get_range() const override
            {
                return range;
            }

            const std::string moduleName;
            const Range range;
        };

        using ModuleDeclarationType = std::variant<NestedModule, Expression, Open>;

    public:
        ModuleDeclaration(ModuleDeclarationType&& declaration);

        [[nodiscard]] Range get_range() const override
        {
            return INodeAlternative::get_range(declaration);
        }

        friend std::string to_string(const ModuleDeclaration& moduleDeclaration);

        const ModuleDeclarationType declaration;
    };

    class Expression : public IASTNode
    {
    public:
        using IExpressionType = INodeAlternative;

        struct Sequential : IExpressionType
        {
            Sequential(
                std::vector<ast_ptr<Expression>>&& expressions,
                bool is_inline,
                Range&& range)
                : expressions(std::move(expressions)),
                  isInline(is_inline),
                  range(range)
            {
            }

            ~Sequential() override = default;

            friend std::string to_string(const Sequential& sequential);

            [[nodiscard]] Range get_range() const override
            {
                return range;
            }

            const std::vector<ast_ptr<Expression>> expressions;
            const bool isInline;
            const Range range;
        };

        struct Append : IExpressionType
        {
            Append(std::vector<ast_ptr<Expression>>&& expressions, const Range&& range)
                : expressions(std::move(expressions)),
                  range(range)
            {
            }

            friend std::string to_string(const Append& append);

            [[nodiscard]] Range get_range() const override
            {
                return range;
            }

            const std::vector<ast_ptr<Expression>> expressions;
            const Range range;
        };

        struct Tuple : IExpressionType
        {
            Tuple(std::vector<ast_ptr<Expression>>&& expressions, const Range&& range)
                : expressions(std::move(expressions)),
                  range(range)
            {
            }

            friend std::string to_string(const Tuple& tuple);

            [[nodiscard]] Range get_range() const override
            {
                return range;
            }

            const std::vector<ast_ptr<Expression>> expressions;
            const Range range;
        };

        struct OP : IExpressionType
        {
            enum class Type
            {
                LOGICAL,
                EQUALITY,
                RELATION,
                ARITHMETIC
            };

            enum class LogicalType
            {
                AND,
                OR
            };

            enum class EqualityType
            {
                EQUAL,
                NOT_EQUAL
            };

            enum class RelationType
            {
                LESS,
                GREATER,
                LESS_EQUAL,
                GREATER_EQUAL
            };

            enum class ArithmeticType
            {
                ADD,
                SUBTRACT,
                MULTIPLY,
                DIVIDE,
                MODULO
            };

            using Operators = std::variant<
                std::vector<LogicalType>,
                std::vector<EqualityType>,
                std::vector<RelationType>,
                std::vector<ArithmeticType>>;

            OP(std::vector<ast_ptr<Expression>>&& expressions,
               OP::Type type,
               Operators&& st,
               Range&& range)
                : expressions(std::move(expressions)),
                  type(type),
                  ops(std::move(st)),
                  range(range)
            {
            }

            friend std::string to_string(const OP& or_expr);

            [[nodiscard]] Range get_range() const override
            {
                return range;
            }

            const std::vector<ast_ptr<Expression>> expressions;
            const OP::Type type;
            const Operators ops;
            const Range range;
        };

        struct DotGet : IExpressionType
        {
            DotGet(
                ast_ptr<Expression>&& expression,
                const std::string& identifier,
                const Range&& range)
                : expression(std::move(expression)),
                  identifier(identifier),
                  range(range)
            {
            }

            friend std::string to_string(const DotGet& dot_get);

            [[nodiscard]] Range get_range() const override
            {
                return range;
            }

            const ast_ptr<Expression> expression;
            const std::string identifier;
            const Range range;
        };

        struct DotIndexedGet : IExpressionType
        {
            DotIndexedGet(
                ast_ptr<Expression>&& base_expression,
                ast_ptr<Expression>&& index_expression,
                const Range&& range)
                : base_expression(std::move(base_expression)),
                  index_expression(std::move(index_expression)),
                  range(range)
            {
            }

            friend std::string to_string(const DotIndexedGet& dot_get);

            [[nodiscard]] Range get_range() const override
            {
                return range;
            }

            const ast_ptr<Expression> base_expression;
            const ast_ptr<Expression> index_expression;
            const Range range;
        };

        struct Typed final : IExpressionType
        {
            Typed(const ast_ptr<Expression>& expression, const ast_ptr<Type>& type, const Range& range)
                : expression(expression),
                  type(type),
                  range(range)
            {
            }

            friend std::string to_string(const Typed& typed);
            [[nodiscard]] Range get_range() const override { return range; }

            const ast_ptr<Expression> expression;
            const ast_ptr<Type> type;
            const Range range;
        };

        struct Unary final : IExpressionType
        {
            enum class Type
            {
                PLUS,
                MINUS,
                NOT
            };

            Unary(ast_ptr<Expression>&& expression, Type type, Range&& range)
                : expression(std::move(expression)),
                  type(type),
                  range(range)
            {
            }

            friend std::string to_string(const Unary& unary);

            [[nodiscard]] Range get_range() const override { return range; }

            const ast_ptr<Expression> expression;
            const Type type;
            const Range range;
        };

        struct Paren : IExpressionType
        {
            Paren(ast_ptr<Expression>&& expression, const Range&& range)
                : expression(std::move(expression)),
                  range(range)
            {
            }

            friend std::string to_string(const Paren& paren);

            [[nodiscard]] Range get_range() const override { return range; }

            const ast_ptr<Expression> expression;
            const Range range;
        };

        struct Constant : IExpressionType
        {
            Constant(ast_ptr<fsharpgrammar::Constant>&& constant)
                : constant(std::move(constant))
            {}

            friend std::string to_string(const Expression::Constant& constant);
            [[nodiscard]] Range get_range() const override { return constant->get_range(); }

            const ast_ptr<fsharpgrammar::Constant> constant;
        };

        struct Ident : IExpressionType
        {
            Ident(ast_ptr<fsharpgrammar::Ident>&& ident)
                : ident(std::move(ident))
            {}

            friend std::string to_string(const Ident& ident) { return utils::to_string(*ident.ident); }
            [[nodiscard]] Range get_range() const override { return ident->get_range(); }

            const ast_ptr<fsharpgrammar::Ident> ident;
        };

        struct LongIdent : IExpressionType
        {
            LongIdent(ast_ptr<fsharpgrammar::LongIdent>&& longIdent)
                : longIdent(std::move(longIdent))
            {}

            friend std::string to_string(const LongIdent& ident) { return utils::to_string(*ident.longIdent); }
            [[nodiscard]] Range get_range() const override { return longIdent->get_range(); }

            const ast_ptr<fsharpgrammar::LongIdent> longIdent;
        };

        struct Match : IExpressionType
        {
            Match(
                ast_ptr<Expression>&& expression,
                std::vector<ast_ptr<MatchClause>>&& clauses,
                Range&& range)
                : expression(std::move(expression)),
                  clauses(std::move(clauses)),
                  range(range)
            {}

            friend std::string to_string(const Match& match);
            [[nodiscard]] Range get_range() const override { return range; }

            const ast_ptr<Expression> expression;
            const std::vector<ast_ptr<MatchClause>> clauses;
            const Range range;
        };


        using ExpressionType = std::variant<
            Sequential,
            Append,
            Tuple,
            OP,
            DotGet,
            DotIndexedGet,
            Typed,
            Unary,
            Paren,
            Constant,
            Ident,
            LongIdent,
            Match,
            PlaceholderNodeAlternative>;

    public:
        explicit Expression(ExpressionType&& expression);

        friend std::string to_string(const Expression& expression)
        {
            return utils::to_string(expression.expression);
        }

        [[nodiscard]] Range get_range() const override;

        const ExpressionType expression;
    };

    class MatchClause final : public IASTNode
    {
    public:
        MatchClause(ast_ptr<Pattern>&& pattern,
            std::optional<ast_ptr<Expression>>&& when_expression,
            std::vector<ast_ptr<Expression>>&& expressions,
            Range&& range)
            : pattern(std::move(pattern)),
              when_expression(std::move(when_expression)),
              expressions(std::move(expressions)),
              range(range)
        {}

        friend std::string to_string(const MatchClause& match_clause);
        [[nodiscard]] Range get_range() const override { return range; }

    public:
        const ast_ptr<Pattern> pattern;
        const std::optional<ast_ptr<Expression>> when_expression;
        const std::vector<ast_ptr<Expression>> expressions;
        const Range range;
    };

    class Pattern : public IASTNode
    {
    public:
        using IPatternType = INodeAlternative;

        struct TuplePattern : IPatternType
        {
            TuplePattern(std::vector<ast_ptr<Pattern>>&& patterns, const Range&& range)
                : patterns(std::move(patterns)),
                  range(range)
            {
            }

            friend std::string to_string(const TuplePattern& tuplePattern);

            [[nodiscard]] Range get_range() const override
            {
                return range;
            }

            const std::vector<ast_ptr<Pattern>> patterns;
            const Range range;
        };

        using PatternType = std::variant<TuplePattern>;

    public:
        explicit Pattern(PatternType&& pattern);

        friend std::string to_string(const Pattern& pattern)
        {
            return utils::to_string(pattern.pattern);
        }

        [[nodiscard]] Range get_range() const override;

        const PatternType pattern;
    };

    class Type final : public IASTNode
    {
    public:
        explicit Type(const Range& range)
            : range(range)
        {
        }

        friend std::string to_string(const Type& type)
        {
            return "Type " + utils::to_string(type.range);
        }

        [[nodiscard]] Range get_range() const override { return range; }

        const Range range;
    };

} // fsharpmlir
