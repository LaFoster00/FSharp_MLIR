project(compiler)

set(LLVM_LINK_COMPONENTS
        Support
)

add_subdirectory(include)
add_subdirectory(lib)

add_library(FSharpCompiler STATIC
        mlir/ASTToMLIR.cpp
        mlir/SetupDependencies.cpp
)

add_dependencies(FSharpCompiler
        MLIRFSharpDialect
)

get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(translation_libs GLOBAL PROPERTY MLIR_TRANSLATION_LIBS)

target_link_libraries(FSharpCompiler
        PUBLIC
        ${dialect_libs}
        ${conversion_libs}

        FSharpGrammar
)

#include all the necessary files
set(COMPILER_INCLUDE_DIRECTORIES
        ${LLVM_INCLUDE_DIRS}
        ${MLIR_INCLUDE_DIRS}
        ${PROJECT_SOURCE_DIR}/include
        ${PROJECT_BINARY_DIR}/include
        CACHE INTERNAL ""
)

target_include_directories(FSharpCompiler PUBLIC
        ${COMPILER_INCLUDE_DIRECTORIES}
)
include_directories(${COMPILER_INCLUDE_DIRECTORIES})
link_directories(${LLVM_BUILD_LIBRARY_DIR})
add_definitions(${LLVM_DEFINITIONS})

add_subdirectory(test)