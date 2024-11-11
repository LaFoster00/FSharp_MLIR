set(LLVM_TARGET_DEFINITIONS FooOps.td)
mlir_tablegen(FooOps.h.inc -gen-op-decls)
mlir_tablegen(FooOps.cpp.inc -gen-op-defs)
mlir_tablegen(FooTypes.h.inc -gen-typedef-decls -typedefs-dialect=foo)
mlir_tablegen(FooTypes.cpp.inc -gen-typedef-defs -typedefs-dialect=foo)
mlir_tablegen(FooDialect.h.inc -gen-dialect-decls -dialect=foo)
mlir_tablegen(FooDialect.cpp.inc -gen-dialect-defs -dialect=foo)
add_public_tablegen_target(MLIRFooOpsIncGen)
add_dependencies(mlir-headers MLIRFooOpsIncGen)

add_mlir_doc(FooDialect FooDialect Foo/ -gen-dialect-doc)

add_mlir_dialect_library(MLIRFoo
        ../FooDialect.cpp

        ADDITIONAL_HEADER_DIRS
        ${PROJECT_SOURCE_DIR}/lib/Dialect
)