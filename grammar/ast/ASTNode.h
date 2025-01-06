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
        {
        }

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
            std::optional<ast_ptr<LongIdent>> name,
            std::vector<ast_ptr<ModuleDeclaration>>&& module_decls,
            Range&& range);
        ~ModuleOrNamespace() override = default;

        [[nodiscard]] Range get_range() const override
        {
            return range;
        }

        friend std::string to_string(const ModuleOrNamespace& moduleOrNamespace);

        const Type type;
        const std::optional<ast_ptr<LongIdent>> name;
        const std::vector<ast_ptr<ModuleDeclaration>> moduleDecls;
        const Range range;
    };

    class ModuleDeclaration : public IASTNode
    {
    public:
        struct NestedModule : INodeAlternative
        {
            NestedModule(
                ast_ptr<LongIdent> name,
                std::vector<ast_ptr<ModuleDeclaration>>&& module_decls,
                Range&& range);

            friend std::string to_string(const NestedModule& nestedModuleDeclaration);

            [[nodiscard]] Range get_range() const override
            {
                return range;
            }

            const ast_ptr<LongIdent> name;
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
            Open(ast_ptr<LongIdent> module_name, Range&& range);

            friend std::string to_string(const Open& open)
            {
                return "Open " + utils::to_string(*open.moduleName);
            }

            [[nodiscard]] Range get_range() const override
            {
                return range;
            }

            const ast_ptr<LongIdent> moduleName;
            const Range range;
        };

        using ModuleDeclarationType = std::variant<NestedModule, Expression, Open>;

    public:
        explicit ModuleDeclaration(ModuleDeclarationType&& declaration);

        [[nodiscard]] Range get_range() const override
        {
            return INodeAlternative::get_range(declaration);
        }

        friend std::string to_string(const ModuleDeclaration& moduleDeclaration);

        const ModuleDeclarationType declaration;
    };

    class Expression final : public IASTNode
    {
    public:
        using IExpressionType = INodeAlternative;

        struct Sequential final : IExpressionType
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

        struct Append final : IExpressionType
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

        struct Tuple final : IExpressionType
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

        struct OP final : IExpressionType
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

        struct DotGet final : IExpressionType
        {
            DotGet(
                ast_ptr<Expression>&& expression,
                ast_ptr<LongIdent>&& identifier,
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
            const ast_ptr<LongIdent> identifier;
            const Range range;
        };

        struct DotIndexedGet final : IExpressionType
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

        struct Paren final : IExpressionType
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

        struct Constant final : IExpressionType
        {
            explicit Constant(ast_ptr<fsharpgrammar::Constant>&& constant)
                : constant(std::move(constant))
            {
            }

            friend std::string to_string(const Expression::Constant& constant);
            [[nodiscard]] Range get_range() const override { return constant->get_range(); }

            const ast_ptr<fsharpgrammar::Constant> constant;
        };

        struct Ident final : IExpressionType
        {
            explicit Ident(ast_ptr<fsharpgrammar::Ident>&& ident)
                : ident(std::move(ident))
            {
            }

            friend std::string to_string(const Ident& ident) { return utils::to_string(*ident.ident); }
            [[nodiscard]] Range get_range() const override { return ident->get_range(); }

            const ast_ptr<fsharpgrammar::Ident> ident;
        };

        struct LongIdent final : IExpressionType
        {
            explicit LongIdent(ast_ptr<fsharpgrammar::LongIdent>&& longIdent)
                : longIdent(std::move(longIdent))
            {
            }

            friend std::string to_string(const LongIdent& ident) { return utils::to_string(*ident.longIdent); }
            [[nodiscard]] Range get_range() const override { return longIdent->get_range(); }

            const ast_ptr<fsharpgrammar::LongIdent> longIdent;
        };

        struct Null final : IExpressionType
        {
            explicit Null(Range&& range) : range(range)
            {
            }

            friend std::string to_string(const Null& null) { return "Null"; }
            [[nodiscard]] Range get_range() const override { return range; }

            const Range range;
        };

        struct Record final : IExpressionType
        {
            struct Field
            {
                ast_ptr<fsharpgrammar::Ident> ident;
                ast_ptr<Expression> expression;
            };

            Record(std::vector<Field>&& fields, Range&& range)
                : fields(std::move(fields)),
                  range(range)
            {
            }

            friend std::string to_string(const Record& record);
            [[nodiscard]] Range get_range() const override { return range; }

            const std::vector<Field> fields;
            const Range range;
        };

        struct Array final : IExpressionType
        {
            Array(std::vector<ast_ptr<Expression>>&& expressions, Range&& range)
                : expressions(std::move(expressions)), range(range)
            {
            }

            friend std::string to_string(const Array& array);
            [[nodiscard]] Range get_range() const override { return range; }

            const std::vector<ast_ptr<Expression>> expressions;
            const Range range;
        };

        struct List final : IExpressionType
        {
            List(std::vector<ast_ptr<Expression>>&& expressions, Range&& range)
                : expressions(std::move(expressions)), range(range)
            {
            }

            friend std::string to_string(const List& list);
            [[nodiscard]] Range get_range() const override { return range; }

            const std::vector<ast_ptr<Expression>> expressions;
            const Range range;
        };

        struct New final : IExpressionType
        {
            New(ast_ptr<fsharpgrammar::Type>&& type,
                std::optional<ast_ptr<Expression>>&& expression,
                Range&& range)
                : type(std::move(type)), expression(std::move(expression)), range(range)
            {
            }

            friend std::string to_string(const New& n);
            [[nodiscard]] Range get_range() const override { return range; }

            const ast_ptr<fsharpgrammar::Type> type;
            const std::optional<ast_ptr<Expression>> expression;
            const Range range;
        };

        struct IfThenElse final : IExpressionType
        {
            IfThenElse(
                ast_ptr<Expression>&& condition,
                std::vector<ast_ptr<Expression>>&& then,
                std::optional<std::vector<ast_ptr<Expression>>>&& else_expr,
                Range&& range)
                : condition(std::move(condition)),
                  then(std::move(then)),
                  else_expr(std::move(else_expr)),
                  range(range)
            {
            }

            friend std::string to_string(const IfThenElse& if_then_else);
            [[nodiscard]] Range get_range() const override { return range; }

            const ast_ptr<Expression> condition;
            const std::vector<ast_ptr<Expression>> then;
            const std::optional<std::vector<ast_ptr<Expression>>> else_expr;
            const Range range;
        };

        struct Match final : IExpressionType
        {
            Match(
                ast_ptr<Expression>&& expression,
                std::vector<ast_ptr<MatchClause>>&& clauses,
                Range&& range)
                : expression(std::move(expression)),
                  clauses(std::move(clauses)),
                  range(range)
            {
            }

            friend std::string to_string(const Match& match);
            [[nodiscard]] Range get_range() const override { return range; }

            const ast_ptr<Expression> expression;
            const std::vector<ast_ptr<MatchClause>> clauses;
            const Range range;
        };

        struct PipeRight final : IExpressionType
        {
            PipeRight(
                ast_ptr<Expression>&& previous_expression,
                std::vector<ast_ptr<Expression>>&& expressions,
                Range&& range)
                : previous_expression(std::move(previous_expression)),
                  expressions(std::move(expressions)),
                  range(range)
            {
            }

            friend std::string to_string(const PipeRight& right);
            [[nodiscard]] Range get_range() const override { return range; }

            const ast_ptr<Expression> previous_expression;
            const std::vector<ast_ptr<Expression>> expressions;
            const Range range;
        };

        struct Let final : IExpressionType
        {
            Let(const bool is_mutable,
                const bool is_recursive,
                ast_ptr<Pattern>&& args,
                std::vector<ast_ptr<Expression>>&& expressions,
                Range&& range)
                : isMutable(is_mutable),
                  isRecursive(is_recursive),
                  args(std::move(args)),
                  expressions(std::move(expressions)),
                  range(range)
            {
            }

            friend std::string to_string(const Let& let);
            [[nodiscard]] Range get_range() const override { return range; }

            const bool isMutable;
            const bool isRecursive;
            const ast_ptr<Pattern> args;
            const std::vector<ast_ptr<Expression>> expressions;
            const Range range;
        };

        /// F# syntax: ident.ident...ident <- expr
        struct LongIdentSet final : IExpressionType
        {
            LongIdentSet(
                ast_ptr<fsharpgrammar::LongIdent>&& long_ident,
                ast_ptr<Expression>&& expression,
                Range&& range)
                : long_ident(std::move(long_ident)),
                  expression(std::move(expression)),
                  range(range)
            {
            }

            friend std::string to_string(const LongIdentSet& long_ident_set);
            [[nodiscard]] Range get_range() const override { return range; }

            const ast_ptr<fsharpgrammar::LongIdent> long_ident;
            const ast_ptr<Expression> expression;
            const Range range;
        };

        /// F# syntax: expr <- expr
        struct Set final : IExpressionType
        {
            Set(ast_ptr<Expression>&& target_expression,
                ast_ptr<Expression>&& expression,
                Range&& range)
                : target_expression(std::move(target_expression)),
                  expression(std::move(expression)),
                  range(range)
            {
            }

            friend std::string to_string(const Set& set);
            [[nodiscard]] Range get_range() const override { return range; }

            const ast_ptr<Expression> target_expression;
            const ast_ptr<Expression> expression;
            const Range range;
        };

        /// F# syntax: expr.ident...ident <- expr
        struct DotSet final : IExpressionType
        {
            DotSet(ast_ptr<Expression>&& target_expression,
                   ast_ptr<fsharpgrammar::LongIdent>&& long_ident,
                   ast_ptr<Expression>&& expression,
                   Range&& range)
                : target_expression(std::move(target_expression)),
                  long_ident(std::move(long_ident)),
                  expression(std::move(expression)),
                  range(range)
            {
            }

            friend std::string to_string(const DotSet& dot_set);
            [[nodiscard]] Range get_range() const override { return range; }

            const ast_ptr<Expression> target_expression;
            const ast_ptr<fsharpgrammar::LongIdent> long_ident;
            const ast_ptr<Expression> expression;
            const Range range;
        };

        /// F# syntax: expr.[expr, ..., expr] <- expr
        struct DotIndexSet final : IExpressionType
        {
            DotIndexSet(ast_ptr<Expression>&& pre_bracket_expression,
                        std::vector<ast_ptr<Expression>>&& bracket_expressions,
                        ast_ptr<Expression>&& expression,
                        Range&& range)
                : pre_bracket_expression(std::move(pre_bracket_expression)),
                  bracket_expressions(std::move(bracket_expressions)),
                  expression(std::move(expression)),
                  range(range)
            {
            }

            friend std::string to_string(const DotIndexSet& dot_index_set);
            [[nodiscard]] Range get_range() const override { return range; }

            const ast_ptr<Expression> pre_bracket_expression;
            const std::vector<ast_ptr<Expression>> bracket_expressions;
            const ast_ptr<Expression> expression;
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
            Null,
            Record,
            Array,
            List,
            New,
            IfThenElse,
            Match,
            PipeRight,
            Let,
            LongIdentSet,
            Set,
            DotSet,
            DotIndexSet,
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
        {
        }

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

        using PatternType = std::variant<
            TuplePattern,
            PlaceholderNodeAlternative
        >;

    public:
        explicit Pattern(PatternType&& pattern);

        friend std::string to_string(const Pattern& pattern)
        {
            return utils::to_string(pattern.pattern);
        }

        [[nodiscard]] Range get_range() const override { return INodeAlternative::get_range(pattern); }

        const PatternType pattern;
    };

    class Type final : public IASTNode
    {
    public:
        using ITypeType = INodeAlternative;

        struct Fun final : ITypeType
        {
            Fun(ast_ptr<Type>&& left,
                std::vector<ast_ptr<Type>>&& fun_types,
                Range&& range)
                : left(std::move(left)),
                  fun_types(std::move(fun_types)),
                  range(range)
            {
            }

            friend std::string to_string(const Type::Fun& type);
            [[nodiscard]] Range get_range() const override { return range; }

            const ast_ptr<Type> left;
            const std::vector<ast_ptr<Type>> fun_types;
            const Range range;
        };

        struct Tuple final : ITypeType
        {
            Tuple(std::vector<ast_ptr<Type>>&& types,
                  Range&& range)
                : types(std::move(types)),
                  range(range)
            {
            }

            friend std::string to_string(const Tuple& tuple);
            [[nodiscard]] Range get_range() const override { return range; };

            const std::vector<ast_ptr<Type>>& types;
            const Range range;
        };

        struct Postfix final : ITypeType
        {
            Postfix(ast_ptr<Type>&& left,
                    ast_ptr<Type>&& right,
                    bool is_paren,
                    Range&& range)
                : left(std::move(left)),
                  right(std::move(right)),
                  is_paren(is_paren),
                  range(range)
            {
            }

            friend std::string to_string(const Postfix& postfix);
            [[nodiscard]] Range get_range() const override { return range; }

            const ast_ptr<Type> left;
            const ast_ptr<Type> right;
            const bool is_paren;
            const Range range;
        };

        struct Array final : ITypeType
        {
            Array(ast_ptr<Type>&& type, Range&& range)
                : type(std::move(type)),
                  range(range)
            {
            }

            friend std::string to_string(const Array& array);
            [[nodiscard]] Range get_range() const override { return range; }

            const ast_ptr<Type> type;
            const Range range;
        };

        struct Paren final : ITypeType
        {
            Paren(ast_ptr<Type>&& type,
                  Range&& range)
                : type(std::move(type)),
                  range(range)
            {
            }

            friend std::string to_string(const Paren& parent);
            [[nodiscard]] Range get_range() const override { return range; }

            const ast_ptr<Type> type;
            const Range range;
        };

        struct Var final : ITypeType
        {
            Var(ast_ptr<Ident>&& ident,
                Range&& range)
                : ident(std::move(ident)),
                  range(range)
            {
            }

            friend std::string to_string(const Var& var);
            [[nodiscard]] Range get_range() const override { return range; }

            const ast_ptr<Ident> ident;
            const Range range;
        };

        struct LongIdent final : ITypeType
        {
            LongIdent(ast_ptr<fsharpgrammar::LongIdent>&& long_ident,
                      Range&& range)
                : longIdent(std::move(long_ident)),
                  range(range)
            {
            }

            friend std::string to_string(const LongIdent& ident);
            [[nodiscard]] Range get_range() const override { return range; }

            const ast_ptr<fsharpgrammar::LongIdent> longIdent;
            const Range range;
        };

        struct Anon final : ITypeType
        {
            explicit Anon(Range&& range)
                : range(range)
            {
            }

            friend std::string to_string(const Anon& anon);
            [[nodiscard]] Range get_range() const override { return range; }

            const Range range;
        };

        struct StaticConstant final : ITypeType
        {
            StaticConstant(ast_ptr<Constant>&& constant,
                           Range&& range)
                : constant(std::move(constant)),
                  range(range)
            {
            }

            friend std::string to_string(const StaticConstant& constant);
            [[nodiscard]] Range get_range() const override { return range; }

            ast_ptr<Constant> constant;
            const Range range;
        };

        struct StaticNull final : ITypeType
        {
            explicit StaticNull(Range&& range)
                : range(range)
            {
            }

            friend std::string to_string(const StaticNull& null);
            [[nodiscard]] Range get_range() const override { return range; }

            const Range range;
        };

        using TypeVariant = std::variant<
            Fun,
            Tuple,
            Postfix,
            Array,
            Paren,
            Var,
            LongIdent,
            Anon,
            StaticConstant,
            StaticNull,
            PlaceholderNodeAlternative
        >;

    public:
        explicit Type(TypeVariant&& type)
            : type(std::move(type))
        {
        }

        friend std::string to_string(const Type& type)
        {
            return utils::to_string(type.type);
        }

        [[nodiscard]] Range get_range() const override { return INodeAlternative::get_range(type); }
        const TypeVariant type;
    };
} // fsharpmlir
