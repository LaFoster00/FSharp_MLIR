project(compiler)

add_subdirectory(include)
add_subdirectory(lib)

include_directories(
        ${PROJECT_SOURCE_DIR}/include
        ${PROJECT_BINARY_DIR}/include
)

add_subdirectory(test)