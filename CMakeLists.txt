cmake_minimum_required(VERSION 3.20)

# Check if the operating system is supported
if (NOT UNIX)
    message(WARNING "Unsupported operating system")
endif ()

# Check if the compiler is supported, by matching against a regex pattern containing the supported compiler clang
if (NOT CMAKE_CXX_COMPILER MATCHES ".*clang.*")
    message(WARNING "Unsupported compiler")
endif ()

if(POLICY CMP0068)
    cmake_policy(SET CMP0068 NEW)
    set(CMAKE_BUILD_WITH_INSTALL_NAME_DIR ON)
endif()

if(POLICY CMP0075)
    cmake_policy(SET CMP0075 NEW)
endif()

if(POLICY CMP0077)
    cmake_policy(SET CMP0077 NEW)
endif()

# LLVM requires CMP0116 for tblgen: https://reviews.llvm.org/D101083
# CMP0116: Ninja generators transform `DEPFILE`s from `add_custom_command()`
# New in CMake 3.20. https://cmake.org/cmake/help/latest/policy/CMP0116.html
if(POLICY CMP0116)
    cmake_policy(SET CMP0116 OLD)
endif()

if(POLICY CMP0135)
    cmake_policy(SET CMP0135 NEW)
endif()

if (CMAKE_BUILD_TYPE STREQUAL "Debug")
    add_compile_definitions(DEBUG)
endif()

set(CMAKE_EXPORT_COMPILE_COMMANDS=ON)

set(CMAKE_CXX_STANDARD 20 CACHE STRING "C++ standard to conform to" FORCE)
set(CMAKE_CXX_STANDARD_REQUIRED YES)

set(CMAKE_BUILD_WITH_INSTALL_NAME_DIR ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
list(APPEND CMAKE_PREFIX_PATH "extern/llvm-project/mlir/cmake")
list(APPEND CMAKE_PREFIX_PATH "extern/magic_enum")

include(FetchContent)
include(CMakePackageConfigHelpers)

message(STATUS "CMAKE_INSTALL_PREFIX ${CMAKE_INSTALL_PREFIX}")
message(STATUS "CMAKE_INSTALL_INCLUDEDIR ${CMAKE_INSTALL_INCLUDEDIR}")
message(STATUS "CMAKE_INSTALL_LIBDIR ${CMAKE_INSTALL_LIBDIR}")
message(STATUS "CMAKE_INSTALL_BINDIR ${CMAKE_INSTALL_BINDIR}")

project(
    FSharp_MLIR
    VERSION 0.1
    LANGUAGES CXX C
)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -frtti")

# CMAKE UTILITIES
include(cmake/utils.cmake)

# Boost Setup
include(cmake/boost.cmake)

# LLVM Setup
include(cmake/llvm.cmake)

find_package(MLIR REQUIRED CONFIG)

message(STATUS "Using MLIRConfig.cmake in: ${MLIR_DIR}")
message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")

set(LLVM_RUNTIME_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/bin)
set(LLVM_LIBRARY_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/lib)

list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")

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

# cpptrace setup
include(cmake/cpptrace.cmake)

# gtest setup
include(cmake/GTest.cmake)

# Add Project source files
add_subdirectory(utils)
add_subdirectory(grammar)
add_subdirectory(compiler)
