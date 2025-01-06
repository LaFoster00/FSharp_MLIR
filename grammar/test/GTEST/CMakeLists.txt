enable_testing()

add_executable(GrammarTests
    tests.cpp
)

target_include_directories(GrammarTests
        PRIVATE
        ${PROJECT_SOURCE_DIR}/../
        ${FSHARP_MLIR_UTILITY_INCLUDES})

target_link_libraries(GrammarTests
        PRIVATE
        GTest::gtest_main

        FSharpGrammar
        ${FSHARP_MLIR_UTILITY_LIBS}
)

include(GoogleTest)
gtest_discover_tests(GrammarTests)