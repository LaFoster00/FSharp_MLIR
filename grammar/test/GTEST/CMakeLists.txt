enable_testing()

add_executable(FSharpGrammarTests
    tests.cpp
)

target_link_libraries(FSharpGrammarTests
        PRIVATE
        GTest::gtest_main

        PUBLIC
        FSharpGrammar
)

include(GoogleTest)
gtest_discover_tests(FSharpGrammarTests)