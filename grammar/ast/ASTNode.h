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
    template<typename T>
    class IASTNodeT;
    template <typename T>
    using ast_ptr = std::shared_ptr<T>;

    template <typename T,typename... Args>
    requires std::is_base_of_v<IASTNodeT<T>, T>
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
        [[nodiscard]] virtual std::string get_string() const = 0;
    };

    template<typename T>
    class IASTNodeT : public IASTNode
    {
    public:
        ~IASTNodeT() override = default;

        [[nodiscard]] std::string get_string() const override
        {
            return to_string(*static_cast<const T*>(this));
        }
    };

    struct INodeAlternative
    {
        virtual ~INodeAlternative() = default;
        [[nodiscard]] virtual Range get_range() const = 0;
        [[nodiscard]] virtual std::string get_string() const = 0;
    };

    template<typename T>
    struct INodeAlternativeT : public INodeAlternative
    {
        virtual ~INodeAlternativeT() = default;

        std::string get_string() const override
        {
            return to_string(*static_cast<const T*>(this));
        }
    };
}

template<typename T>
    struct fmt::formatter<T, std::enable_if_t<std::is_base_of_v<fsharpgrammar::IASTNodeT<T>, T>, char>> : fmt::formatter<std::string>
{
    auto format(const fsharpgrammar::IASTNode& node, fmt::format_context& ctx) const
    {
        return fmt::formatter<std::string>::format(node.get_string(), ctx);
    }
};

namespace fsharpgrammar
{

    class ModuleOrNamespace;
    class ModuleDeclaration;
    class Expression;

    class Main : public IASTNodeT<Main>
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

        friend std::string to_string(const Main& main)
        {
            return "Main";
        }

    private:
        std::vector<ast_ptr<ModuleOrNamespace>> modules_or_namespaces;
        const Range range;
    };

    class ModuleOrNamespace : public IASTNodeT<ModuleOrNamespace>
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

    class ModuleDeclaration : public IASTNodeT<ModuleDeclaration>
    {
    public:
        struct NestedModule : INodeAlternativeT<NestedModule>
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

        struct Expression : INodeAlternativeT<Expression>
        {
            Expression(
                ast_ptr<fsharpgrammar::Expression>&& expression,
                Range&& range);

            friend std::string to_string(const Expression& expression)
            {
                return "Expression";
            }

            [[nodiscard]] Range get_range() const override
            {
                return range;
            }

            const ast_ptr<fsharpgrammar::Expression> expression;
            const Range range;
        };

        struct Open : INodeAlternativeT<Open>
        {
            Open(const std::string& module_name, Range&& range);

            friend std::string to_string(const Open& open)
            {
                return "Open";
            }

            [[nodiscard]] Range get_range() const override
            {
                return range;
            }

            const std::string moduleName;
            const Range range;
        };
    public:
        ModuleDeclaration(std::unique_ptr<INodeAlternative> &&declaration);

        [[nodiscard]] Range get_range() const override
        {
            return declaration->get_range();
        }

        friend std::string to_string(const ModuleDeclaration& moduleDeclaration);

        const std::unique_ptr<INodeAlternative> declaration;
    };

    class Expression : public IASTNodeT<Expression>
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

            [[nodiscard]] std::string get_string() const override
            {
                return "Sequential";
            }
            [[nodiscard]] Range get_range() const override
            {
                return range;
            }

            const std::vector<ast_ptr<Expression>> expressions;
            const bool isInline;
            const Range range;
        };

    public:
        Expression(std::unique_ptr<IExpressionType> &&expression);

        friend std::string to_string(const Expression& expression)
        {
            return expression.get_string();
        }

        [[nodiscard]] Range get_range() const override
        {
            return expression->get_range();
        }

        [[nodiscard]] std::string get_string() const override
        {
            return to_string(*this);
        }

        std::unique_ptr<IExpressionType> expression;
    };

    class Pattern : public IASTNodeT<Expression>
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

        [[nodiscard]] std::string get_string() const override
        {
            return "Pattern";
        }

        const Type type;
        const std::optional<std::string> value;
        const Range range;
    };
} // fsharpmlir
