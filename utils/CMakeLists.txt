project(Utils)

# Find all .cpp files in the current directory
file(GLOB SOURCE "${PROJECT_SOURCE_DIR}/utils/*.cpp" "${PROJECT_SOURCE_DIR}/utils/*.h")

add_library(Utils STATIC
        ${SOURCE}
)

target_include_directories(Utils PUBLIC
        ${PROJECT_SOURCE_DIR}/Utils
)

append_to_cached_variable(FSHARP_MLIR_UTILITY_LIBS
        Utils
)

append_to_cached_variable(FSHARP_MLIR_UTILITY_INCLUDES
        ${PROJECT_SOURCE_DIR}
)