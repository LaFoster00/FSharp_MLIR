project(Test)

set(LLVM_LINK_COMPONENTS
        Support
)

# add generated grammar to demo binary target
add_executable(FSharpGrammarTimedTest
        main.cpp
)

target_link_libraries(FSharpGrammarTimedTest
        PRIVATE
        FSharpGrammar
)


# Copy all .fs files to build directory
file(GLOB FS_FILES "${CMAKE_CURRENT_SOURCE_DIR}/FSharpReferenceProject/*.fs")
foreach(FS_FILE ${FS_FILES})
    configure_file(${FS_FILE} ${CMAKE_CURRENT_BINARY_DIR} COPYONLY)
endforeach()

add_subdirectory(GTEST)