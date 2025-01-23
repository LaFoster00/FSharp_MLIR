enable_testing()

add_executable(FSharpCompilerTest
        ASTToMLIRTest.cpp
)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
target_link_libraries(FSharpCompilerTest
        PRIVATE
        GTest::gtest_main

        PUBLIC
        FSharpGrammar
        FSharpCompiler
)

target_precompile_headers(FSharpCompilerTest REUSE_FROM FSharpCompiler)

include(GoogleTest)
gtest_discover_tests(FSharpCompilerTest)

# Ensure the directory exists
file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/TestFiles)

# Copy all .fs files to build directory
file(GLOB FS_FILES "${CMAKE_CURRENT_SOURCE_DIR}/TestFiles/*.fs")
foreach(FS_FILE ${FS_FILES})
    configure_file(${FS_FILE} ${CMAKE_CURRENT_BINARY_DIR}/TestFiles/ COPYONLY)
endforeach()