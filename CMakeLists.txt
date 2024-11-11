cmake_minimum_required(VERSION 3.25)

set(CMAKE_C_COMPILER clang)
set(CMAKE_CXX_COMPILER clang++)

# LLVM requires CMP0116 for tblgen: https://reviews.llvm.org/D101083
# CMP0116: Ninja generators transform `DEPFILE`s from `add_custom_command()`
# New in CMake 3.20. https://cmake.org/cmake/help/latest/policy/CMP0116.html
set(CMAKE_POLICY_DEFAULT_CMP0116 OLD)
if(POLICY CMP0116)
    cmake_policy(SET CMP0116 OLD)
endif()

set(CMAKE_CXX_STANDARD 20 CACHE STRING "C++ standard to conform to")
set(CMAKE_CXX_STANDARD_REQUIRED YES)

set(CMAKE_BUILD_WITH_INSTALL_NAME_DIR ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
list(APPEND CMAKE_PREFIX_PATH "extern/llvm-project/mlir/cmake")

include(FetchContent)


project(
    FSharp_MLIR
    VERSION 0.1
    LANGUAGES CXX C
)

# LLVM Setup
include(cmake/llvm.cmake)
build_llvm()

find_package(MLIR REQUIRED CONFIG)
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

# ANTLR4 SETUP
include(cmake/antlr4.cmake)
build_antlr()

add_subdirectory(compiler)
add_subdirectory(grammar)

