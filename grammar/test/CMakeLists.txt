project(Test)

set(LLVM_LINK_COMPONENTS
        Support
)

# add generated grammar to demo binary target
add_executable(test
        main.cpp
)

add_dependencies(test FSharpGrammar)

# include generated files in project environment
target_include_directories(test PRIVATE
        ${PROJECT_SOURCE_DIR}/../
)

target_link_libraries(test
        PRIVATE
        FSharpGrammar
        antlr4_shared
)


# Copy all .fs files to build directory
file(GLOB FS_FILES "${CMAKE_CURRENT_SOURCE_DIR}/*.fs")
foreach(FS_FILE ${FS_FILES})
    configure_file(${FS_FILE} ${CMAKE_CURRENT_BINARY_DIR} COPYONLY)
endforeach()