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

target_precompile_headers(FSharpCompilerTest PRIVATE
        ${PROJECT_SOURCE_DIR}/include/compiler/PrecompileHeaders.h)

include(GoogleTest)
gtest_discover_tests(FSharpCompilerTest)

# Ensure the directory exists
file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/TestFiles)

# Copy all .fs files to build directory
file(GLOB FS_FILES "${CMAKE_CURRENT_SOURCE_DIR}/TestFiles/*.fs")
foreach(FS_FILE ${FS_FILES})
    configure_file(${FS_FILE} ${CMAKE_CURRENT_BINARY_DIR}/TestFiles/ COPYONLY)
endforeach()