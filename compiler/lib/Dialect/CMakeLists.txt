add_subdirectory(include)

set(LLVM_LINK_COMPONENTS
        Support
)

add_llvm_executable(dialect_test
        dialect_test.cpp
        FooDialect.cpp

        DEPENDS
        MLIRFooOpsIncGen
)

target_include_directories(dialect_test
        PUBLIC
        include/
        ${CMAKE_CURRENT_BINARY_DIR}/include/
)

target_link_libraries(dialect_test
        PRIVATE
        MLIRAnalysis
        MLIRFunctionInterfaces
        MLIRIR
        MLIRParser
        MLIRSideEffectInterfaces
        MLIRTransforms)