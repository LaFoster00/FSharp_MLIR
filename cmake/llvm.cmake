macro(build_llvm)
    message(STATUS "Building llvm dependencies.")

    set(_CMAKE_BUILD_TYPE "${CMAKE_BUILD_TYPE}")

    set(LLVM_ENABLE_PROJECTS "" CACHE STRING "")
    list(APPEND LLVM_ENABLE_PROJECTS mlir lld llvm)

    set(LLVM_TARGETS_TO_BUILD "" CACHE STRING "")
    list(APPEND LLVM_TARGETS_TO_BUILD X86 ARM AArch64 RISCV)

    set(LLVM_INCLUDE_EXAMPLES OFF CACHE BOOL "")

    set(LLVM_INCLUDE_TESTS OFF CACHE BOOL "")

    set(LLVM_INCLUDE_BENCHMARKS OFF CACHE BOOL "")

    set(LLVM_ENABLE_IDE ON CACHE BOOL "")

    set(LLVM_ENABLE_LLD ON CACHE BOOL "")

    set(LLVM_CCACHE_BUILD ON CACHE BOOL "")

    set(LLVM_INSTALL_UTILS ON CACHE BOOL "")

    message(VERBOSE "Building LLVM Targets: ${LLVM_TARGETS_TO_BUILD}")
    message(VERBOSE "Building LLVM Projects: ${LLVM_ENABLE_PROJECTS}")

    set(LLVM_LIBRARY_OUTPUT_INTDIR "${CMAKE_CURRENT_BINARY_DIR}/llvm-project/lib")
    set(LLVM_RUNTIME_OUTPUT_INTDIR "${CMAKE_CURRENT_BINARY_DIR}/llvm-project/bin")

    message(STATUS "Configuring extern/llvm-project")
    list(APPEND CMAKE_MESSAGE_INDENT "  ")
    set(LLVM_CMAKE_SOURCE_SUBDIR "extern/llvm-project/llvm")
    add_subdirectory("${LLVM_CMAKE_SOURCE_SUBDIR}" "llvm-project" EXCLUDE_FROM_ALL)
    get_directory_property(LLVM_VERSION_MAJOR DIRECTORY "${LLVM_CMAKE_SOURCE_SUBDIR}" LLVM_VERSION_MAJOR)
    if (NOT LLVM_VERSION_MAJOR)
        message(SEND_ERROR "Failed to read LLVM_VERSION_MAJOR property on LLVM directory. Should have been set since https://github.com/llvm/llvm-project/pull/83346.")
    endif()

    list(POP_BACK CMAKE_MESSAGE_INDENT)

    set(CMAKE_BUILD_TYPE "${_CMAKE_BUILD_TYPE}" )
    list(APPEND CMAKE_PREFIX_PATH "${CMAKE_CURRENT_BINARY_DIR}/lib/cmake/mlir")
endmacro()