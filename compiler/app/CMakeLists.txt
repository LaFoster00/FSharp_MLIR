
add_executable(FSharpCompilerApp
        FSharpCompilerApp.cpp
)

# Link statically to reduce dependencies
# This will make the binary larger, but it will be easier to distribute
target_link_libraries(FSharpCompilerApp
        PRIVATE
        FSharpCompiler
)

target_include_directories(FSharpCompilerApp PRIVATE ${PROJECT_SOURCE_DIR}/include)