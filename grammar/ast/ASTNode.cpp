//
// Created by lasse on 12/18/24.
//

#include "ASTNode.h"

namespace fsharpgrammar
{
    ModuleOrNamespace::ModuleOrNamespace(
        Type type,
        std::optional<std::string> name,
        Range&& range)
        :
        type(type),
        name(name),
        range(range)
    {
    }

    Main::Main(std::vector<ModuleOrNamespace>& modules_or_namespaces, Range&& range)
        :
        modules_or_namespaces(std::move(modules_or_namespaces)),
        range(range)
    {
    }
} // fsharpgrammar
