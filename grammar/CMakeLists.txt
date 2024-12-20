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

add_library(FSharpGrammar INTERFACE)
target_link_libraries(FSharpGrammar
    INTERFACE
        FSharpGrammarAntlr
        FSharpGrammarAST
        antlr4_shared
)
target_include_directories(FSharpGrammar INTERFACE
    ${PROJECT_SOURCE_DIR}
)

add_subdirectory(test)
