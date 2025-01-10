add_mlir_dialect_library(MLIRFSharpDialect
        FSharpDialect.cpp

        ADDITIONAL_HEADER_DIRS
        ${PROJECT_SOURCE_DIR}/include/compiler

        DEPENDS
        MLIRFSharpOpsIncGen

        LINK_LIBS PUBLIC
        MLIRIR
        MLIRInferTypeOpInterface
        MLIRFuncDialect
)
target_include_directories(MLIRFSharpDialect PRIVATE
        ${COMPILER_INCLUDE_DIRECTORIES}
)