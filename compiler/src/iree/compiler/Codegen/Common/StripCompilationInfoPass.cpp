// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Common/Passes.h"
#include "mlir/Transforms/WalkPatternRewriteDriver.h"

#define DEBUG_TYPE "iree-codegen-strip-compilation-info"

namespace mlir::iree_compiler {

#define GEN_PASS_DEF_STRIPCOMPILATIONINFOPASS
#include "iree/compiler/Codegen/Common/Passes.h.inc"

namespace {

// Checks whether the funcOp has any compilation info.
bool hasCompilationInfo(func::FuncOp funcOp) {
  if (getTranslationInfo(funcOp))
    return true;

  bool hasAttrConfig = false;
  funcOp.walk([&](Operation *op) {
    if (getCompilationInfo(op) || getLoweringConfig(op)) {
      hasAttrConfig = true;
      return;
    }
  });

  // Return success if any relevant attributes were found.
  return hasAttrConfig;
}

struct StripCompilationInfo : public OpRewritePattern<func::FuncOp> {
  using OpRewritePattern<func::FuncOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(func::FuncOp funcOp,
                                PatternRewriter &rewriter) const final {
    if (!hasCompilationInfo(funcOp))
      return failure();

    // if the function has translation info, erase it
    if (getTranslationInfo(funcOp)) {
      eraseTranslationInfo(funcOp);
    }

    funcOp->walk([&](Operation *op) {
      if (getCompilationInfo(op)) {
        // Erase the compilation info configuration if it exists
        eraseCompilationInfo(op);
      }
      if (getLoweringConfig(op)) {
        // Erase the lowering configuration from root operation if it
        // exists.
        eraseLoweringConfig(op);
      }
    });

    return success();
  }
};

struct StripCompilationInfoPass final
    : impl::StripCompilationInfoPassBase<StripCompilationInfoPass> {
  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    patterns.add<StripCompilationInfo>(&getContext());
    walkAndApplyPatterns(getOperation(), std::move(patterns));
  }
};
} // namespace
} // namespace mlir::iree_compiler
