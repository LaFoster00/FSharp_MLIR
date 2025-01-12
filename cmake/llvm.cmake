message(STATUS "Setting up LLVM dependencies.")

# Define the paths
set(LLVM_SOURCE_DIR "${PROJECT_SOURCE_DIR}/extern/llvm-project/llvm")
set(LLVM_BUILD_DIR "${CMAKE_BINARY_DIR}/extern/llvm-project/build-${CMAKE_BUILD_TYPE}")
set(LLVM_INSTALL_DIR "${LLVM_BUILD_DIR}/install") # Optional, depending on use case

message(STATUS "LLVM_SOURCE_DIR: ${LLVM_SOURCE_DIR}")
message(STATUS "LLVM_BUILD_DIR: ${LLVM_BUILD_DIR}")
message(STATUS "LLVM_INSTALL_DIR: ${LLVM_INSTALL_DIR}")

# Option to force build
option(FORCE_BUILD_LLVM "Force build LLVM project" OFF)

# Create the build directory if it doesn't exist
file(MAKE_DIRECTORY ${LLVM_BUILD_DIR})

list(APPEND CMAKE_MESSAGE_INDENT "  ")

# Check if the build target exists
if(NOT EXISTS "${LLVM_BUILD_DIR}/bin/llvm-config" OR ${FORCE_BUILD_LLVM})
    message(STATUS "LLVM build target does not exist. Building.")

    list(APPEND CMAKE_MESSAGE_INDENT "  ")
    # Run the CMake configuration command for LLVM
    execute_process(
            COMMAND ${CMAKE_COMMAND} -G Ninja ${LLVM_SOURCE_DIR}
            -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
            -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}
            -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}
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
else()
    message(STATUS "LLVM build target already exists. Skipping build.")
endif()

# Update paths for find_package to locate MLIR
set(MLIR_DIR "${LLVM_BUILD_DIR}/lib/cmake/mlir" CACHE INTERNAL "")
set(LLVM_DIR "${LLVM_BUILD_DIR}/lib/cmake/llvm" CACHE INTERNAL "")
list(APPEND CMAKE_PREFIX_PATH "${MLIR_DIR}" "${LLVM_DIR}")
message(STATUS "Updated CMAKE_PREFIX_PATH for MLIR and LLVM")