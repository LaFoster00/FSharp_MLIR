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

    template <typename T,typename... Args>
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

        template<typename... T>
        requires are_all_base_of<INodeAlternative, T...>
        static Range get_range(std::variant<T...> alternatives)
        {
            return std::visit([](const auto& obj) {
                        return obj.get_range();
                    }, alternatives);
        }

        template<typename... T>
        requires are_all_base_of<INodeAlternative, T...>
        friend std::string to_string(const std::variant<T...> &alternatives)
        {
            return std::visit([](const auto& obj) {
                        return utils::to_string(obj);
                    }, alternatives);
        }
    };



}

template<typename T>
    struct fmt::formatter<T, std::enable_if_t<std::is_base_of_v<fsharpgrammar::IASTNode, T>, char>> : fmt::formatter<std::string>
{
    auto format(const fsharpgrammar::IASTNode& node, fmt::format_context& ctx) const
    {
        return fmt::formatter<std::string>::format(utils::to_string(static_cast<const T&>(node)), ctx);
    }
};

template<typename T>
    struct fmt::formatter<T, std::enable_if_t<std::is_base_of_v<fsharpgrammar::INodeAlternative, T>, char>> : fmt::formatter<std::string>
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
                const std::string& name,
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
            Open(const std::string& module_name, Range&& range);

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
        ModuleDeclaration(ModuleDeclarationType &&declaration);

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

        using ExpressionType = std::variant<Sequential>;
    public:
        Expression(ExpressionType &&expression);

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

    class Pattern : public IASTNode
    {
    public:
        enum class Type
        {
            TuplePattern,
            AndPattern
        };

        Pattern(Type type, std::vector<ast_ptr<Pattern>>&& patterns, Range&& range);
        ~Pattern() override = default;

        [[nodiscard]] Range get_range() const override
        {
            return range;
        }

        friend std::string to_string(const Pattern& pattern);

        const Type type;
        const std::vector<ast_ptr<Pattern>> patterns;
        const Range range;
    };
} // fsharpmlir
