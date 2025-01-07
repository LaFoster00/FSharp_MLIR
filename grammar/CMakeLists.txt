if (NOT UNIX)
    message(WARNING "Unsupported operating system")
endif ()

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

install(TARGETS
        FSharpGrammar
        FSharpGrammarAntlr
        FSharpGrammarAST
        EXPORT GrammarTargets
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
)

install(EXPORT GrammarTargets
        FILE GrammarTargets.cmake
        NAMESPACE Grammar::
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/Grammar
)

add_subdirectory(test)
