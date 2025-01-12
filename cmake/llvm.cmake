
message(STATUS "Building llvm dependencies.")

# Define the paths
set(LLVM_SOURCE_DIR "${PROJECT_SOURCE_DIR}/extern/llvm-project/llvm")
set(LLVM_BUILD_DIR "${LLVM_SOURCE_DIR}/extern/llvm-project/build")
set(LLVM_INSTALL_DIR "${LLVM_BUILD_DIR}/install") # Optional, depending on use case

message("LLVM_SOURCE_DIR: ${LLVM_SOURCE_DIR}")
message("LLVM_BUILD_DIR: ${LLVM_BUILD_DIR}")
message("LLVM_INSTALL_DIR: ${LLVM_INSTALL_DIR}")

# Create the build directory if it doesn't exist
file(MAKE_DIRECTORY ${LLVM_BUILD_DIR})

list(APPEND CMAKE_MESSAGE_INDENT "  ")
# Run the CMake configuration command for LLVM
execute_process(
        COMMAND ${CMAKE_COMMAND} -G Ninja ${LLVM_SOURCE_DIR}
        -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
        -DLLVM_ENABLE_PROJECTS=mlir
        -DLLVM_BUILD_EXAMPLES=OFF
        -DLLVM_TARGETS_TO_BUILD=X86
        -DLLVM_ENABLE_ASSERTIONS=ON
        -DLLVM_ENABLE_LLD=ON
        -DLLVM_CCACHE_BUILD=ON
        -DLLVM_ENABLE_IDE=ON
        -DLLVM_INCLUDE_TESTS=OFF
        -DLLVM_INCLUDE_BENCHMARKS=OFF
        -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
        WORKING_DIRECTORY ${LLVM_BUILD_DIR}
        RESULT_VARIABLE llvm_cmake_result
)
list(POP_BACK CMAKE_MESSAGE_INDENT)

# Check if the CMake configuration succeeded
if(llvm_cmake_result)
    message(FATAL_ERROR "Failed to configure LLVM project in ${LLVM_BUILD_DIR}")
endif()

list(APPEND CMAKE_MESSAGE_INDENT "  ")
# Build the LLVM project
execute_process(
        COMMAND ninja
        WORKING_DIRECTORY ${LLVM_BUILD_DIR}
        RESULT_VARIABLE llvm_build_result
)
list(POP_BACK CMAKE_MESSAGE_INDENT)

# Check if the build succeeded
if(llvm_build_result)
    message(FATAL_ERROR "Failed to build LLVM project in ${LLVM_BUILD_DIR}")
endif()

# Update paths if necessary for find_package to locate MLIR
set(MLIR_DIR "${LLVM_BUILD_DIR}/lib/cmake/mlir" CACHE INTERNAL "")
set(LLVM_DIR "${LLVM_BUILD_DIR}/lib/cmake/llvm" CACHE INTERNAL "")
list(APPEND CMAKE_PREFIX_PATH "${MLIR_DIR}")
list(APPEND CMAKE_PREFIX_PATH "${LLVM_DIR}")