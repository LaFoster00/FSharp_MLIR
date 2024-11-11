project(Test)

set(LLVM_LINK_COMPONENTS
        Support
)

message(STATUS "${SampleGrammarLexer_OUTPUT_DIR}")
message(STATUS "${SampleGrammarParser_OUTPUT_DIR}")
message(STATUS "${SampleGrammarLexer_CXX_OUTPUTS}")
message(STATUS "${SampleGrammarParser_CXX_OUTPUTS}")

# add generated grammar to demo binary target
add_executable(test main.cpp
        ${SampleGrammarLexer_CXX_OUTPUTS}
        ${SampleGrammarParser_CXX_OUTPUTS}
)

add_dependencies(test MLIRFooOpsIncGen SampleGrammarGenerator)

# include generated files in project environment
target_include_directories(test PRIVATE ${SampleGrammarLexer_OUTPUT_DIR})
target_include_directories(test PRIVATE ${SampleGrammarParser_OUTPUT_DIR})
target_include_directories(test PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/include/)

target_link_libraries(test
        PRIVATE
        antlr4_static
        MLIRAnalysis
        MLIRFunctionInterfaces
        MLIRIR
        MLIRParser
        MLIRSideEffectInterfaces
        MLIRTransforms
)