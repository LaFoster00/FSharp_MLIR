# Find all .cpp files in the current directory
file(GLOB SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp" "${CMAKE_CURRENT_SOURCE_DIR}/*.h")

add_library(FSharpAST STATIC
        ${SOURCE}
)

add_dependencies(FSharpAST FSharpGrammarGenerator)

target_link_libraries(FSharpAST
        PRIVATE
        antlr4_shared
        Utils
)

message(STATUS "UTILS_INCLUDE_DIR: ${UTILS_INCLUDE_DIR}")
target_include_directories(FSharpAST PUBLIC
        ${UTILS_INCLUDE_DIR}
)