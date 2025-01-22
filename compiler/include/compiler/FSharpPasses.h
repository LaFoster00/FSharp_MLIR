//
// Created by lasse on 1/14/25.
//

#pragma once

#include <memory>

namespace mlir {
    class Pass;

    namespace fsharp {
        // Create a pass that lowers the fsharp.closure ops to func.func ops since they are the ones we want to use going forward.
        // This will also resolve nested functions and convert them to capturing global functions
        std::unique_ptr<Pass> createLowerToFunctionPass();

        /// Create a pass for lowering operations the remaining `fsharp` operations, as
        /// well as `Affine` and `Std`, to the LLVM dialect for codegen.
        std::unique_ptr<Pass> createLowerToLLVMPass();



    } // namespace toy
} // namespace mlir
