add_library(ASTToMLIR STATIC
        ASTToMLIR.cpp
        SetupDependencies.cpp
)
add_dependencies(ASTToMLIR
        FSharpGrammar
        MLIRFSharpOpsIncGen
)

set_target_properties(ASTToMLIR PROPERTIES
        COMPILE_FLAGS "-frtti"
)

target_link_libraries(ASTToMLIR
        PRIVATE
        MLIRAnalysis
        MLIRFunctionInterfaces
        MLIRIR
        MLIRParser
        MLIRSideEffectInterfaces
        MLIRTransforms
        MLIRArithDialect
        MLIRLLVMDialect
        MLIRFuncDialect
        MLIRAffineDialect


        PUBLIC
        FSharpGrammar
)

target_include_directories(ASTToMLIR PUBLIC
        ${COMPILER_INCLUDE_DIRECTORIES}
)

add_subdirectory(test)