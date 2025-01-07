add_llvm_library(ASTToMLIR
        ASTToMLIR.cpp

        DEPENDS
        FSharpGrammar  # Ensure the AST target is built first
)
llvm_update_compile_flags(ASTToMLIR)

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

        PUBLIC
        FSharpGrammar
)

target_include_directories(ASTToMLIR PUBLIC "$<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}>")

add_subdirectory(test)