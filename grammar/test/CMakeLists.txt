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