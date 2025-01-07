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

add_library(FSharpGrammar STATIC
        Grammar.h
        Grammar.cpp
)
target_link_libraries(FSharpGrammar
    PUBLIC
        FSharpGrammarAntlr
        FSharpGrammarAST
        Utils
)
target_include_directories(FSharpGrammar PUBLIC
    ${PROJECT_SOURCE_DIR}
)

install(TARGETS
        FSharpGrammar
        ARCHIVE DESTINATION lib)

install(FILES)


add_subdirectory(test)
