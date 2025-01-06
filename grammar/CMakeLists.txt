if(NOT UNIX)
    message(WARNING "Unsupported operating system")
endif()

project(
        grammar
        VERSION 0.1
        LANGUAGES CXX C
)

add_subdirectory(antlr)
add_subdirectory(ast)

add_library(FSharpGrammar SHARED
        Grammar.h
        Grammar.cpp
)
target_link_libraries(FSharpGrammar
    PUBLIC
        FSharpGrammarAntlr
        FSharpGrammarAST
        antlr4_shared
        ${FSHARP_MLIR_UTILITY_LIBS}
)
target_include_directories(FSharpGrammar PUBLIC
    ${PROJECT_SOURCE_DIR}
    ${FSHARP_MLIR_UTILITY_INCLUDES}
)

add_subdirectory(test)
