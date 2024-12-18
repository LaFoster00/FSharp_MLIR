project(Utils)

# Find all .cpp files in the current directory
file(GLOB SOURCE "${PROJECT_SOURCE_DIR}/utils/*.cpp" "${PROJECT_SOURCE_DIR}/utils/*.h")

add_library(Utils STATIC
        ${SOURCE}
)

target_include_directories(Utils PRIVATE
        ${PROJECT_SOURCE_DIR}/Utils
)

set(UTILS_INCLUDE_DIR ${PROJECT_SOURCE_DIR} CACHE INTERNAL "")