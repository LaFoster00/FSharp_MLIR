get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
set(LIBS
        ${dialect_libs}
        ${conversion_libs}
        MLIRArithDialect
        MLIROptLib

        MLIRFoo
)

add_llvm_executable(FooTest dialect_test.cpp)
llvm_update_compile_flags(FooTest)

target_link_libraries(FooTest PRIVATE ${LIBS})

mlir_check_all_link_libraries(FooTest)