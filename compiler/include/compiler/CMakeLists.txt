add_mlir_dialect(FSharp fsharp)

set(LLVM_TARGET_DEFINITIONS FSharp.td)
mlir_tablegen(FSharpAttrDefs.h.inc -gen-attrdef-decls
        -attrdefs-dialect=FSharp)
mlir_tablegen(FSharpAttrDefs.cpp.inc -gen-attrdef-defs
        -attrdefs-dialect=FSharp)
add_public_tablegen_target(MLIRFSharpAttrDefsIncGen)
add_dependencies(mlir-headers MLIRFSharpAttrDefsIncGen)