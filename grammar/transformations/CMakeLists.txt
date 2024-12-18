add_library(ASTTransformations STATIC
    ASTCanonicalizer.cpp
)

add_dependencies(ASTTransformations FSharpGrammarGenerator)

target_link_libraries(ASTTransformations
        PRIVATE
        antlr4_shared
)