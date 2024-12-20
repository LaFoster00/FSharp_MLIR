//
// Created by lasse on 12/18/24.
//

#pragma once

#include <optional>
#include <vector>

#include "Range.h"
#include "fmt/core.h"
#include <type_traits>



namespace fsharpgrammar
{
    template <typename T>
    using ast_ptr = std::shared_ptr<T>;

    template <typename T, typename... Args>
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
}

template<typename T>
    struct fmt::formatter<T, std::enable_if_t<std::is_base_of_v<fsharpgrammar::IASTNode, T>, char>> : fmt::formatter<std::string>
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
    class NestedModuleDeclaration;
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

        friend std::string to_string(const Main& main)
        {
            return "Main";
        }

        [[nodiscard]] std::string get_string() const override
        {
            return to_string(*this);
        }

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

        [[nodiscard]] std::string get_string() const override
        {
            return to_string(*this);
        }

        const Type type;
        const std::optional<std::string> name;
        const std::vector<ast_ptr<ModuleDeclaration>> module_decls;
        const Range range;
    };

    class ModuleDeclaration : public IASTNode
    {
    public:
        ModuleDeclaration(
            std::vector<ast_ptr<Expression>>&& expressions, Range&& range);

        [[nodiscard]] Range get_range() const override
        {
            return range;
        }

        friend std::string to_string(const ModuleDeclaration& moduleDeclaration);
        [[nodiscard]] std::string get_string() const override
        {
            return to_string(*this);
        }

        const std::vector<ast_ptr<Expression>> expressions;
        const Range range;
    };

    class NestedModuleDeclaration : public IASTNode
    {
    public:
        NestedModuleDeclaration(
            std::string_view name,
            std::vector<ast_ptr<ModuleDeclaration>>&& module_decls,
            Range&& range);

        friend std::string to_string(const NestedModuleDeclaration& nestedModuleDeclaration);

        [[nodiscard]] Range get_range() const override
        {
            return range;
        }

        [[nodiscard]] std::string get_string() const override
        {
            return to_string(*this);
        }

        const std::string name;
        const std::vector<ast_ptr<ModuleDeclaration>> module_decls;
        const Range range;
    };

    class Expression : public IASTNode
    {
    public:
        Expression(Range&& range);

        friend std::string to_string(const Expression& expression);

        [[nodiscard]] Range get_range() const override
        {
            return range;
        }

        [[nodiscard]] std::string get_string() const override
        {
            return to_string(*this);
        }

        Range range;
    };

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

        [[nodiscard]] std::string get_string() const override
        {
            return "Pattern";
        }

        const Type type;
        const std::optional<std::string> value;
        const Range range;
    };
} // fsharpmlir
