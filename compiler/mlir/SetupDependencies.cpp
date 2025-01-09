//
// Created by lasse on 1/9/25.
//

#include "SetupDependencies.h"

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Builders.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"

using namespace mlir;

namespace fsharpgrammar::compiler
{
    /// Inserts printf so that we can call it from inside the application
      /// Create a function declaration for printf, the signature is:
      ///   * `i32 (i8*, ...)`
    LLVM::LLVMFunctionType getPrintfType(MLIRContext* context)
    {
        auto llvmI32Ty = IntegerType::get(context, 32);
        auto llvmPtrTy = LLVM::LLVMPointerType::get(context);
        auto llvmFnType = LLVM::LLVMFunctionType::get(llvmI32Ty, llvmPtrTy,
                                                      /*isVarArg=*/true);
        return llvmFnType;
    }

    /// Return a symbol reference to the printf function, inserting it into the
    /// module if necessary.
    FlatSymbolRefAttr getOrInsertPrintf(OpBuilder& builder,
                                        ModuleOp module)
    {
        auto* context = module.getContext();
        if (module.lookupSymbol<LLVM::LLVMFuncOp>("printf"))
            return SymbolRefAttr::get(context, "printf");

        // Insert the printf function into the body of the parent module.
        builder.setInsertionPointToStart(module.getBody());
        builder.create<LLVM::LLVMFuncOp>(module.getLoc(), "printf", getPrintfType(context));
        return SymbolRefAttr::get(context, "printf");
    }

    /// Return a value representing an access into a global string with the given
    /// name, creating the string if necessary.
    Value getOrCreateGlobalString(Location loc, OpBuilder& builder,
                                  StringRef name, StringRef value,
                                  ModuleOp module)
    {
        // Create the global at the entry of the module.
        LLVM::GlobalOp global;
        if (!(global = module.lookupSymbol<LLVM::GlobalOp>(name)))
        {
            OpBuilder::InsertionGuard insertGuard(builder);
            builder.setInsertionPointToStart(module.getBody());
            auto type = LLVM::LLVMArrayType::get(
                IntegerType::get(builder.getContext(), 8), value.size());
            global = builder.create<LLVM::GlobalOp>(loc, type, /*isConstant=*/true,
                                                    LLVM::Linkage::Internal, name,
                                                    builder.getStringAttr(value),
                                                    /*alignment=*/0);
        }

        // Get the pointer to the first character in the global string.
        Value globalPtr = builder.create<LLVM::AddressOfOp>(loc, global);
        Value cst0 = builder.create<LLVM::ConstantOp>(loc, builder.getI64Type(),
                                                      builder.getIndexAttr(0));
        return builder.create<LLVM::GEPOp>(
            loc, LLVM::LLVMPointerType::get(builder.getContext()), global.getType(),
            globalPtr, ArrayRef<Value>({cst0, cst0}));
    }
} // namespace
