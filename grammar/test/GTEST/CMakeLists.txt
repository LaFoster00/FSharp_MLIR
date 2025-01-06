enable_testing()

add_executable(GrammarTests
    tests.cpp
)

target_include_directories(GrammarTests
        PRIVATE
        ${PROJECT_SOURCE_DIR}/../
)

target_link_libraries(GrammarTests
        PRIVATE
        GTest::gtest_main

        PUBLIC
        FSharpGrammar
)

include(GoogleTest)
gtest_discover_tests(GrammarTests)