add_mlir_dialect_library(MLIRFSharpDialect
        FSharpDialect.cpp

        ADDITIONAL_HEADER_DIRS
        ${PROJECT_SOURCE_DIR}/include
        ${PROJECT_BINARY_DIR}/include

        DEPENDS
        MLIRFSharpIncGen
        MLIRFSharpAttrDefsIncGen

        LINK_LIBS PUBLIC
        MLIRIR
        MLIRInferTypeOpInterface
        MLIRFuncDialect
        fmt::fmt
)

target_include_directories(MLIRFSharpDialect PRIVATE
        ${PROJECT_SOURCE_DIR}/include
        ${PROJECT_BINARY_DIR}/include
)