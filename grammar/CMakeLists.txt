if(NOT UNIX)
    message(WARNING "Unsupported operating system")
endif()

project(
        grammar
        VERSION 0.1
        LANGUAGES CXX C
)

include_directories(
        ${PROJECT_SOURCE_DIR}
)

ANTLR_TARGET(FSharpLexer FSharpLexer.g4 LEXER
        PACKAGE fsharpgrammar)
include_directories(${FSharpLexer_OUTPUT_DIR})


ANTLR_TARGET(FSharpParser FSharpParser.g4 PARSER
        PACKAGE fsharpgrammar
        DEPENDS_ANTLR FSharpLexer
        COMPILE_FLAGS -lib ${ANTLR_FSharpLexer_OUTPUT_DIR} #Needed so that the token file can be found
)
include_directories(${FSharpParser_OUTPUT_DIR})

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

add_subdirectory(antlr_src)
add_subdirectory(transformations)

add_subdirectory(test)
