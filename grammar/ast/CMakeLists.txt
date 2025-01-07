# Find all .cpp files in the current directory
file(GLOB HEADER "${CMAKE_CURRENT_SOURCE_DIR}/*.h")
file(GLOB SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp")

add_library(FSharpGrammarAST STATIC
        ${SOURCE}
)

add_dependencies(FSharpGrammarAST FSharpGrammarGenerator)

target_link_libraries(FSharpGrammarAST
        PUBLIC
        Utils
        fmt::fmt
        cpptrace::cpptrace

        FSharpGrammarAntlr
)

message(STATUS "UTILS_INCLUDE_DIR: ${UTILS_INCLUDE_DIR}")
target_include_directories(FSharpGrammarAST PUBLIC
        "$<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}>"
)