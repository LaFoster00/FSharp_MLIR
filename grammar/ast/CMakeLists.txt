# Find all .cpp files in the current directory
file(GLOB SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp" "${CMAKE_CURRENT_SOURCE_DIR}/*.h")

add_library(FSharpGrammarAST STATIC
        ${SOURCE}
)

add_dependencies(FSharpGrammarAST FSharpGrammarGenerator)

target_link_libraries(FSharpGrammarAST
        PRIVATE
        FSharpGrammarAntlr
        ${FSHARP_MLIR_UTILITY_LIBS}
)

message(STATUS "UTILS_INCLUDE_DIR: ${UTILS_INCLUDE_DIR}")
target_include_directories(FSharpGrammarAST PRIVATE
        ${FSHARP_MLIR_UTILITY_INCLUDES}
)