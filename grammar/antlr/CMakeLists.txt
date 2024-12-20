include_directories(
        ${PROJECT_SOURCE_DIR}
)

ANTLR_TARGET(FSharpLexer FSharpLexer.g4 LEXER
        PACKAGE fsharpgrammar)

ANTLR_TARGET(FSharpParser FSharpParser.g4 PARSER VISITOR
        PACKAGE fsharpgrammar
        DEPENDS_ANTLR FSharpLexer
        COMPILE_FLAGS -lib ${ANTLR_FSharpLexer_OUTPUT_DIR} #Needed so that the token file can be found
)

set(GENERATED_SOURCE
        ${FSharpLexer_CXX_OUTPUTS}
        ${FSharpParser_CXX_OUTPUTS}
)

foreach(src_file ${GENERATED_SOURCE})
    set_source_files_properties(
            ${src_file}
            PROPERTIES
            GENERATED TRUE
    )
endforeach(src_file ${GENERATED_SOURCE})

add_custom_target(FSharpGrammarGenerator DEPENDS ${GENERATED_SOURCE})

message(STATUS "${FSharpLexer_OUTPUT_DIR}")
message(STATUS "${FSharpParser_OUTPUT_DIR}")
message(STATUS "${FSharpLexer_CXX_OUTPUTS}")
message(STATUS "${FSharpParser_CXX_OUTPUTS}")

# add generated grammar to demo binary target
add_library(FSharpGrammarAntlr STATIC
        ${FSharpLexer_CXX_OUTPUTS}
        ${FSharpParser_CXX_OUTPUTS}
        FSharpLexerBase.h
        FSharpLexerBase.cpp
        FSharpParserBase.h
        FSharpParserBase.cpp
)

add_dependencies(FSharpGrammarAntlr
        FSharpGrammarGenerator
)

target_link_libraries(FSharpGrammarAntlr
        PUBLIC
        antlr4_shared
)

target_include_directories(FSharpGrammarAntlr
        PUBLIC
        ${FSharpLexer_OUTPUT_DIR}
        ${FSharpParser_OUTPUT_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}
)