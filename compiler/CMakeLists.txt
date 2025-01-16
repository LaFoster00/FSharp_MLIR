project(compiler)

set(LLVM_LINK_COMPONENTS
        Support
)

add_subdirectory(include)
add_subdirectory(lib)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -frtti")

add_library(FSharpCompiler STATIC
        mlir/Compiler.cpp
        mlir/ASTToMLIR.cpp
        mlir/LowerToLLVM.cpp
)

add_dependencies(FSharpCompiler
        MLIRFSharpDialect
)

get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
get_property(extension_libs GLOBAL PROPERTY MLIR_EXTENSION_LIBS)

target_link_libraries(FSharpCompiler
        PUBLIC
        ${dialect_libs}
        ${conversion_libs}
        ${extension_libs}
        MLIRAnalysis
        MLIRBuiltinToLLVMIRTranslation
        MLIRCallInterfaces
        MLIRCastInterfaces
        MLIRExecutionEngine
        MLIRFunctionInterfaces
        MLIRIR
        MLIRLLVMCommonConversion
        MLIRLLVMToLLVMIRTranslation
        MLIRMemRefDialect
        MLIRParser
        MLIRPass
        MLIRSideEffectInterfaces
        MLIRTargetLLVMIRExport
        MLIRTransforms
        MLIRSupport

        LLVMCore
        LLVMSupport
        LLVMCodeGen
        LLVMOrcJIT

        FSharpGrammar
)

#include all the necessary files
set(COMPILER_INCLUDE_DIRECTORIES
        ${LLVM_INCLUDE_DIRS}
        ${MLIR_INCLUDE_DIRS}
        ${PROJECT_SOURCE_DIR}/include
        ${PROJECT_BINARY_DIR}/include
        CACHE INTERNAL ""
)

target_include_directories(FSharpCompiler PUBLIC
        ${COMPILER_INCLUDE_DIRECTORIES}
)
include_directories(${COMPILER_INCLUDE_DIRECTORIES})
link_directories(${LLVM_BUILD_LIBRARY_DIR})
add_definitions(${LLVM_DEFINITIONS})

add_subdirectory(test)