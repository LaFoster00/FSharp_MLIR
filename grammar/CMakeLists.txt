if(NOT UNIX)
    message(WARNING "Unsupported operating system")
endif()

project(
        grammar
        VERSION 0.1
        LANGUAGES CXX C
)

antlr_target(SampleGrammarLexer TLexer.g4 LEXER
        PACKAGE antlrcpptest)

antlr_target(SampleGrammarParser TParser.g4 PARSER
        PACKAGE antlrcpptest
        DEPENDS_ANTLR SampleGrammarLexer
        COMPILE_FLAGS -lib ${ANTLR_SampleGrammarLexer_OUTPUT_DIR})

# include generated files in project environment
include_directories(${ANTLR_SampleGrammarLexer_OUTPUT_DIR})
include_directories(${ANTLR_SampleGrammarParser_OUTPUT_DIR})

# add generated grammar to demo binary target
add_executable(demo main.cpp
        ${ANTLR_SampleGrammarLexer_CXX_OUTPUTS}
        ${ANTLR_SampleGrammarParser_CXX_OUTPUTS})
target_link_libraries(demo antlr4_static)