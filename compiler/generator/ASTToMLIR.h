//
// Created by lasse on 1/7/25.
//
#pragma once

#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/BuiltinOps.h"

extern mlir::ModuleOp generateMLIRFromAST(std::string_view source, mlir::MLIRContext& context);
