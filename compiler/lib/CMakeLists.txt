add_mlir_dialect_library(MLIRStandalone
        FSharpTypes.cpp
        FSharpDialect.cpp
        FSharpOps.cpp
        FSharpPasses.cpp

        ADDITIONAL_HEADER_DIRS
        ${PROJECT_SOURCE_DIR}/include/compiler

        DEPENDS
        MLIRFSharpOpsIncGen

        LINK_LIBS PUBLIC
        MLIRIR
        MLIRInferTypeOpInterface
        MLIRFuncDialect
)
