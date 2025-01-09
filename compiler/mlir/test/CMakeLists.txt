enable_testing()

add_executable(ASTToMLIRTest
        ASTToMLIRTest.cpp
)

target_link_libraries(ASTToMLIRTest
        PRIVATE
        GTest::gtest_main

        PUBLIC
        FSharpGrammar
        ASTToMLIR
)

target_include_directories(ASTToMLIRTest PUBLIC
        ${COMPILER_INCLUDE_DIRECTORIES}
)

include(GoogleTest)
gtest_discover_tests(ASTToMLIRTest)