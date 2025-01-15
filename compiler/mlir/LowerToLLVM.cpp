//
// Created by lasse on 1/14/25.
//

#include "compiler/FSharpPasses.h"

#include <mlir/Dialect/MemRef/IR/MemRef.h>

#include "compiler/FSharpDialect.h"

#include <mlir/IR/PatternMatch.h>

#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/FunctionCallUtils.h"

using namespace mlir;

/// Generic print function lookupOrCreate helper.
LLVM::LLVMFuncOp lookupOrCreateFn(PatternRewriter& rewriter,
                                  ModuleOp moduleOp,
                                  StringRef name,
                                  ArrayRef<Type> paramTypes,
                                  Type resultType, bool isVarArg)
{
    assert(moduleOp->hasTrait<OpTrait::SymbolTable>() &&
        "expected SymbolTable operation");
    auto func = llvm::dyn_cast_or_null<LLVM::LLVMFuncOp>(
        SymbolTable::lookupSymbolIn(moduleOp, name));
    if (func)
        return func;
    return rewriter.create<LLVM::LLVMFuncOp>(
        moduleOp->getLoc(), name,
        LLVM::LLVMFunctionType::get(resultType, paramTypes, isVarArg));
}