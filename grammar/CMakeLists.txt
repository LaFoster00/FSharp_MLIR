project(
        grammar
        VERSION 0.1
        LANGUAGES CXX C
)

add_subdirectory(antlr)
add_subdirectory(ast)

add_library(FSharpGrammar STATIC
        Grammar.h
        Grammar.cpp
)

target_link_libraries(FSharpGrammar
        PUBLIC
        magic_enum::magic_enum

        FSharpGrammarAntlr
        FSharpGrammarAST
)

target_include_directories(FSharpGrammar PUBLIC
        "$<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}>"
)

add_subdirectory(test)
