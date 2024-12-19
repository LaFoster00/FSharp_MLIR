//
// Created by lasse on 12/18/24.
//

#pragma once

#include <optional>
#include <vector>

#include "Range.h"

namespace fsharpgrammar
{
    class IASTNode
    {
    public:
        friend class ASTBuilder;
        virtual ~IASTNode() = default;

        [[nodiscard]] virtual Range get_range() const = 0;
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
        ModuleOrNamespace(Type type, std::optional<std::string> name, Range&& range);
        ~ModuleOrNamespace() override = default;

        [[nodiscard]] Range get_range() const override
        {
            return range;
        }

        friend std::string to_string(const ModuleOrNamespace& moduleOrNamespace)
        {
            switch (moduleOrNamespace.type)
            {
                case Type::NamedModule:
                    return "Module " + moduleOrNamespace.name.value();
                case Type::AnonymousModule:
                    return "AnonymousModule";
                case Type::Namespace:
                    return "Namespace " + moduleOrNamespace.name.value();
            default:
                return "";
            }
        }
    public:
        const Type type;
        const std::optional<std::string> name;
        const Range range;
    };

    class Main : public IASTNode
    {
    public:
        Main(std::vector<ModuleOrNamespace>& anon_modules, Range&& range);
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
        std::vector<ModuleOrNamespace> modules_or_namespaces;
        const Range range;
    };

    class ModuleDecleration : public IASTNode
    {

    };
} // fsharpmlir
