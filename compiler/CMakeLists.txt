project(compiler)

set(LLVM_LINK_COMPONENTS
        Support
)

add_subdirectory(include)
add_subdirectory(lib)
add_subdirectory(mlir)

#include all the necessary files
set(COMPILER_SOURCE_DIR ${PROJECT_SOURCE_DIR})
set(COMPILER_BINARY_DIR ${PROJECT_BINARY_DIR})
set(COMPILER_INCLUDE_DIRECTORIES
        ${LLVM_INCLUDE_DIRS}
        ${MLIR_INCLUDE_DIRS}
        ${COMPILER_SOURCE_DIR}/include
        ${COMPILER_BINARY_DIR}/include
        CACHE INTERNAL ""
)
include_directories(${COMPILER_INCLUDE_DIRECTORIES})
link_directories(${LLVM_BUILD_LIBRARY_DIR})
add_definitions(${LLVM_DEFINITIONS})