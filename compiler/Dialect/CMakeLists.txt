add_subdirectory(include)

message(STATUS "Including this stupid header ${PROJECT_SOURCE_DIR}/Dialect/include")

add_mlir_dialect_library(MLIRFoo
        PARTIAL_SOURCES_INTENDED
        src/FooDialect.cpp

        ADDITIONAL_HEADER_DIRS
        ${PROJECT_SOURCE_DIR}/Dialect/include

        DEPENDS
        MLIRFooDialectIncGen

        LINK_LIBS PUBLIC
        MLIRIR
        MLIRInferTypeOpInterface
)

add_llvm_executable(dialect_test dialect_test.cpp PARTIAL_SOURCES_INTENDED)
target_link_libraries(dialect_test PUBLIC MLIRFoo)