if(NOT UNIX)
    message(WARNING "Unsupported operating system")
endif()

project(
        grammar
        VERSION 0.1
        LANGUAGES CXX C
)

ANTLR_TARGET(SampleGrammarLexer TLexer.g4 LEXER
        PACKAGE antlrcpptest)

ANTLR_TARGET(SampleGrammarParser TParser.g4 PARSER
        PACKAGE antlrcpptest
        DEPENDS_ANTLR SampleGrammarLexer
        COMPILE_FLAGS -lib ${ANTLR_SampleGrammarLexer_OUTPUT_DIR} #Needed so that the token file can be found
)

set(GENERATED_SOURCE
        ${SampleGrammarLexer_CXX_OUTPUTS}
        ${SampleGrammarParser_CXX_OUTPUTS}
)

foreach(src_file ${GENERATED_SOURCE})
    set_source_files_properties(
            ${src_file}
            PROPERTIES
            GENERATED TRUE
    )
endforeach(src_file ${GENERATED_SOURCE})

add_custom_target(SampleGrammarGenerator DEPENDS ${GENERATED_SOURCE})

