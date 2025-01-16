//
// Created by lasse on 1/14/25.
//

#pragma once

#include <memory>

namespace mlir {
    class Pass;

    namespace fsharp {
        /// Create a pass for lowering operations the remaining `Toy` operations, as
        /// well as `Affine` and `Std`, to the LLVM dialect for codegen.
        std::unique_ptr<Pass> createLowerToLLVMPass();

    } // namespace toy
} // namespace mlir
