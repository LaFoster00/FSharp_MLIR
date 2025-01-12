message(STATUS "Setting up LLVM dependencies.")

# Define the paths
set(LLVM_SOURCE_DIR "${PROJECT_SOURCE_DIR}/extern/llvm-project/llvm")
set(LLVM_BUILD_DIR "${PROJECT_SOURCE_DIR}/extern/llvm-project/build")
set(LLVM_INSTALL_DIR "${LLVM_BUILD_DIR}/install") # Optional, depending on use case

message(STATUS "LLVM_SOURCE_DIR: ${LLVM_SOURCE_DIR}")
message(STATUS "LLVM_BUILD_DIR: ${LLVM_BUILD_DIR}")
message(STATUS "LLVM_INSTALL_DIR: ${LLVM_INSTALL_DIR}")

# Option to force build
option(FORCE_BUILD_LLVM "Force build LLVM project" OFF)

# Function to read CMakeCache.txt and extract CMAKE_BUILD_TYPE
function(get_cached_build_type build_dir result_var)
    set(cache_file "${build_dir}/CMakeCache.txt")
    if(EXISTS ${cache_file})
        file(READ ${cache_file} cache_content)
        string(REGEX MATCH "CMAKE_BUILD_TYPE:STRING=([^\n]+)" match ${cache_content})
        set(${result_var} ${CMAKE_MATCH_1} PARENT_SCOPE)
    else()
        set(${result_var} "" PARENT_SCOPE)
    endif()
endfunction()

# Get the cached build type
get_cached_build_type(${LLVM_BUILD_DIR} CACHED_BUILD_TYPE)
message(STATUS "Cached CMAKE_BUILD_TYPE: ${CACHED_BUILD_TYPE}")
message(STATUS "Current CMAKE_BUILD_TYPE: ${CMAKE_BUILD_TYPE}")


# Create a custom target to configure and build LLVM
add_custom_target(build_llvm_dependencies ALL
        COMMENT "Configuring and building LLVM dependencies"
)

# Create the build directory if it doesn't exist
add_custom_command(
        TARGET build_llvm_dependencies PRE_BUILD
        COMMAND ${CMAKE_COMMAND} -E make_directory ${LLVM_BUILD_DIR}
        COMMENT "Ensuring LLVM build directory exists"
)

# Check if the build target exists
if(NOT EXISTS "${LLVM_BUILD_DIR}/bin/llvm-config" OR ${FORCE_BUILD_LLVM} OR NOT ${CMAKE_BUILD_TYPE} STREQUAL ${CACHED_BUILD_TYPE})
    # Run the CMake configuration command for LLVM
    add_custom_command(
            TARGET build_llvm_dependencies
            PRE_BUILD
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
            COMMENT "Configuring LLVM project"
    )

    # Build the LLVM project
    add_custom_command(
            TARGET build_llvm_dependencies
            POST_BUILD
            COMMAND ninja
            WORKING_DIRECTORY ${LLVM_BUILD_DIR}
            COMMENT "Building LLVM project"
    )
else()
    message(STATUS "LLVM build target already exists. Skipping build.")
endif()

# Update paths for find_package to locate MLIR
set(MLIR_DIR "${LLVM_BUILD_DIR}/lib/cmake/mlir" CACHE INTERNAL "")
set(LLVM_DIR "${LLVM_BUILD_DIR}/lib/cmake/llvm" CACHE INTERNAL "")
list(APPEND CMAKE_PREFIX_PATH "${MLIR_DIR}" "${LLVM_DIR}")
message(STATUS "Updated CMAKE_PREFIX_PATH for MLIR and LLVM")