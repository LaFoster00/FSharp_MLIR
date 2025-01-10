project(Utils)

# Find all .cpp and header files in the current directory
file(GLOB SOURCE "${PROJECT_SOURCE_DIR}/utils/*.cpp")

add_library(Utils STATIC
        ${SOURCE}
)

target_include_directories(Utils PUBLIC
        "$<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>"
)