enable_testing()

add_executable(FSharpCompilerTest
        ASTToMLIRTest.cpp
)

target_link_libraries(FSharpCompilerTest
        PRIVATE
        GTest::gtest_main

        PUBLIC
        FSharpGrammar
        FSharpCompiler
)

include(GoogleTest)
gtest_discover_tests(FSharpCompilerTest)