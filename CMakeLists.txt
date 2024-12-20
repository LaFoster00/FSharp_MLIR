cmake_minimum_required(VERSION 3.25)

# LLVM requires CMP0116 for tblgen: https://reviews.llvm.org/D101083
# CMP0116: Ninja generators transform `DEPFILE`s from `add_custom_command()`
# New in CMake 3.20. https://cmake.org/cmake/help/latest/policy/CMP0116.html
set(CMAKE_POLICY_DEFAULT_CMP0116 OLD)
if(POLICY CMP0116)
    cmake_policy(SET CMP0116 OLD)
endif()

if (CMAKE_BUILD_TYPE STREQUAL "Debug")
    add_compile_definitions(DEBUG)
endif()

set(CMAKE_CXX_STANDARD 20 CACHE STRING "C++ standard to conform to")
set(CMAKE_CXX_STANDARD_REQUIRED YES)

set(CMAKE_BUILD_WITH_INSTALL_NAME_DIR ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
list(APPEND CMAKE_PREFIX_PATH "extern/llvm-project/mlir/cmake")
list(APPEND CMAKE_PREFIX_PATH "extern/magic_enum")

include(FetchContent)

project(
    FSharp_MLIR
    VERSION 0.1
    LANGUAGES CXX C
)

# CMAKE UTILITIES
include(cmake/utils.cmake)

# LLVM Setup
include(cmake/llvm.cmake)
build_llvm()

find_package(MLIR REQUIRED CONFIG)
find_package(LLVM REQUIRED CONFIG)
message(STATUS "Using MLIRConfig.cmake in: ${MLIR_DIR}")
list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")

include(TableGen)
include(AddLLVM)
include(AddMLIR)
include(HandleLLVMOptions)

include_directories(${LLVM_INCLUDE_DIRS}
                    ${MLIR_INCLUDE_DIRS})
link_directories(${LLVM_BUILD_LIBRARY_DIR})
add_definitions(${LLVM_DEFINITIONS})

# Magic enum setup
include(cmake/magic_enum.cmake)

# fmt setup
include(cmake/fmt.cmake)

# ANTLR4 SETUP
include(cmake/antlr4.cmake)
build_antlr()
# add antlr4cpp artifacts to project environment
include_directories(${ANTLR4_INCLUDE_DIRS})
link_libraries(${ANTLR4_RUNTIME_LIBRARIES})

# Define global utility library resources
set(FSHARP_MLIR_UTILITY_LIBS
        magic_enum::magic_enum
        fmt::fmt
        CACHE INTERNAL "")
set(FSHARP_MLIR_UTILITY_INCLUDES CACHE INTERNAL "")

# Add Project source files
add_subdirectory(utils)
add_subdirectory(grammar)
add_subdirectory(compiler)

message(STATUS "FSHARP_MLIR_UTILITY_LIBS : ${FSHARP_MLIR_UTILITY_LIBS}")
message(STATUS "FSHARP_MLIR_UTILITY_INCLUDES : ${FSHARP_MLIR_UTILITY_INCLUDES}")
