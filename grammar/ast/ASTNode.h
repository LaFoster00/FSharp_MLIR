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

            OP(std::vector<ast_ptr<Expression>>&& expressions, Type type, Operators&& st,
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
            const Type type;
            const Operators ops;
            const Range range;
        };

        using ExpressionType = std::variant<Sequential, Append, Tuple, OP, PlaceholderNodeAlternative>;

    public:
        explicit Expression(ExpressionType&& expression);

        friend std::string to_string(const Expression& expression)
        {
            return utils::to_string(expression.expression);
        }

        [[nodiscard]] Range get_range() const override
        {
            return INodeAlternative::get_range(expression);
        }

        const ExpressionType expression;
    };

    std::string to_string(const Expression::Append& append);

    class Pattern : public IASTNode
    {
        enum class Type
        {
            ParenPattern,
            AnonymousExpression,
            Constant,
            NamedPattern,
            RecordPattern,
            ArrayPattern,
            LongIdentifierPattern,
            NullPattern
        };

        Pattern(Type type, std::optional<std::string> value, Range&& range);
        ~Pattern() override = default;

        [[nodiscard]] Range get_range() const override
        {
            return range;
        }

        friend std::string to_string(const Pattern& pattern)
        {
            return "Pattern";
        }

        const Type type;
        const std::optional<std::string> value;
        const Range range;
    };
} // fsharpmlir
