add_mlir_dialect(FSharp fsharp)

set(LLVM_TARGET_DEFINITIONS FSharp.td)
mlir_tablegen(FSharpAttrDefs.h.inc -gen-attrdef-decls -attrdefs-dialect=fsharp)
mlir_tablegen(FSharpAttrDefs.cpp.inc -gen-attrdef-defs -attrdefs-dialect=fsharp)
add_public_tablegen_target(MLIRFSharpAttrDefsIncGen)
add_dependencies(mlir-headers MLIRFSharpAttrDefsIncGen)