message(STATUS "${FSharpLexer_OUTPUT_DIR}")
message(STATUS "${FSharpParser_OUTPUT_DIR}")
message(STATUS "${FSharpLexer_CXX_OUTPUTS}")
message(STATUS "${FSharpParser_CXX_OUTPUTS}")

# add generated grammar to demo binary target
add_library(FSharpGrammar STATIC
        ${FSharpLexer_CXX_OUTPUTS}
        ${FSharpParser_CXX_OUTPUTS}
        FSharpLexerBase.h
        FSharpLexerBase.cpp
        FSharpParserBase.h
        FSharpParserBase.cpp
        ../ast/AstBuilder.cpp
        ../ast/AstBuilder.h
        ../ast/range.cpp
        ../ast/range.h
)

add_dependencies(FSharpGrammar
        FSharpGrammarGenerator
)

target_link_libraries(FSharpGrammar
        PRIVATE
        antlr4_shared
        FSharpAST
)