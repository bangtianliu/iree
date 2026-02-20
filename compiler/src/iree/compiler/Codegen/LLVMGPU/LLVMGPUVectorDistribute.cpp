// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Common/GPU/GPUPatterns.h"
#include "iree/compiler/Codegen/Common/GPU/GPUVectorDistribution.h"
#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenAttrs.h"
#include "iree/compiler/Codegen/Dialect/VectorExt/IR/VectorExtDialect.h"
#include "iree/compiler/Codegen/Dialect/VectorExt/IR/VectorExtOps.h"
#include "iree/compiler/Codegen/LLVMGPU/Passes.h"
#include "iree/compiler/Codegen/LLVMGPU/Utils/LLVMGPUUtils.h"
#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/Dialect/AMDGPU/IR/AMDGPUDialect.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/TypeUtilities.h"

#define DEBUG_TYPE "iree-llvmgpu-vector-distribute"

namespace mlir::iree_compiler {

using VectorValue = TypedValue<VectorType>;

#define GEN_PASS_DEF_LLVMGPUVECTORDISTRIBUTEPASS
#include "iree/compiler/Codegen/LLVMGPU/Passes.h.inc"

ContractionVectorLayoutOptions::ContractionVectorLayoutOptions(
    Operation *root, Value laneId, int64_t subgroupSize,
    ArrayRef<int64_t> workgroupSize)
    : VectorLayoutOptions(root), patterns(root->getContext()) {
  populateGPUDistributionPatterns(patterns);
  populateGPUDistributeNestedLayoutAttrPatterns(patterns, laneId, subgroupSize,
                                                workgroupSize);
  populateGPUDistributeNestedLayoutContractAMDGPUPatterns(patterns);
}

RewritePatternSet &ContractionVectorLayoutOptions::getPatterns() {
  return patterns;
}

VectorLayoutInterface
ContractionVectorLayoutOptions::getDefaultLayout(VectorType type) const {
  // We only allow a default layout for 0-d vectors for now.
  if (type.getRank() > 0) {
    return VectorLayoutInterface();
  }
  ArrayRef<int64_t> empty = {};
  return IREE::VectorExt::NestedLayoutAttr::get(
      type.getContext(), empty, empty, empty, empty, empty, empty, empty);
}

namespace {

/// Set layout anchors for iree_vector_ext.arg_compare operations.
/// This creates to_layout ops wrapping the vector inputs based on the
/// lowering config, enabling the layout analysis to propagate layouts.
static LogicalResult setArgCompareAnchors(mlir::FunctionOpInterface funcOp,
                                          IRRewriter &rewriter,
                                          int64_t subgroupSize) {
  fprintf(stderr, "\n========== setArgCompareAnchors: Starting ==========\n");
  fflush(stderr);
  SmallVector<IREE::VectorExt::ArgCompareOp> argCompareOps;
  funcOp->walk([&](IREE::VectorExt::ArgCompareOp op) {
    argCompareOps.push_back(op);
    fprintf(stderr, "  Found arg_compare op\n");
    fflush(stderr);
  });

  fprintf(stderr, "  Total arg_compare ops found: %zu\n", argCompareOps.size());
  fflush(stderr);

  if (argCompareOps.empty()) {
    fprintf(stderr, "  No arg_compare ops found - returning success\n");
    fflush(stderr);
    return success();
  }

  for (auto argCompareOp : argCompareOps) {
    fprintf(stderr, "  Processing arg_compare op\n");
    fflush(stderr);
    // Get the reduction dimension
    int64_t reductionDim = argCompareOp.getDimension();
    fprintf(stderr, "    Reduction dimension: %lld\n", (long long)reductionDim);
    fflush(stderr);

    // Get vector types for inputs
    TypedValue<VectorType> inputValue = argCompareOp.getInputValue();
    auto inputValueType = inputValue.getType();

    if (!inputValueType) {
      continue; // Skip non-vector arg_compare
    }

    // Create nested layout for the input vector
    // The layout distributes the reduction dimension across threads
    SmallVector<int64_t> shape(inputValueType.getShape().begin(),
                                inputValueType.getShape().end());
    int64_t rank = shape.size();

    // Initialize layout dimensions (subgroup, batch, outer, thread, element)
    SmallVector<int64_t> subgroupTile(rank, 1);
    SmallVector<int64_t> batchTile(rank, 1);
    SmallVector<int64_t> outerTile(rank, 1);
    SmallVector<int64_t> threadTile(rank, 1);
    SmallVector<int64_t> elementTile(rank, 1);

    // Distribute the reduction dimension across threads (subgroup)
    // If the dimension is larger than subgroupSize, use element_tile for the remainder
    int64_t reductionSize = shape[reductionDim];
    if (reductionSize <= subgroupSize) {
      threadTile[reductionDim] = reductionSize;
    } else {
      threadTile[reductionDim] = subgroupSize;
      // Each thread handles multiple elements
      elementTile[reductionDim] = (reductionSize + subgroupSize - 1) / subgroupSize;
    }

    // Set strides (subgroup and thread)
    SmallVector<int64_t> subgroupStrides(rank, 0);
    SmallVector<int64_t> threadStrides(rank, 0);
    threadStrides[reductionDim] = 1; // Threads stride along reduction dim

    // Create the nested layout attribute
    auto inputLayout = IREE::VectorExt::NestedLayoutAttr::get(
        rewriter.getContext(), subgroupTile, batchTile, outerTile,
        threadTile, elementTile, subgroupStrides, threadStrides);

    // Wrap input value with to_layout
    rewriter.setInsertionPoint(argCompareOp);
    Location loc = argCompareOp.getLoc();

    auto layoutedInputValue = IREE::VectorExt::ToLayoutOp::create(
        rewriter, loc, inputValue, inputLayout);

    // Update arg_compare operand 0 (input value)
    argCompareOp->setOperand(0, layoutedInputValue.getResult());

    // If there's an explicit input index, wrap it too
    if (Value inputIndex = argCompareOp.getInputIndex()) {
      auto inputIndexVec = dyn_cast<TypedValue<VectorType>>(inputIndex);
      if (inputIndexVec) {
        auto layoutedInputIndex = IREE::VectorExt::ToLayoutOp::create(
            rewriter, loc, inputIndexVec, inputLayout);
        argCompareOp->setOperand(1, layoutedInputIndex.getResult());
      }
    }

    // Wrap init value and init index with reduced-rank layout
    TypedValue<VectorType> initValue = argCompareOp.getInitValue();
    TypedValue<VectorType> initIndex = argCompareOp.getInitIndex();

    auto initValueType = initValue.getType();
    if (initValueType.getRank() > 0) {
      SmallVector<int64_t> initShape(initValueType.getShape().begin(),
                                      initValueType.getShape().end());
      int64_t initRank = initShape.size();

      SmallVector<int64_t> initSubgroupTile(initRank, 1);
      SmallVector<int64_t> initBatchTile(initRank, 1);
      SmallVector<int64_t> initOuterTile(initRank, 1);
      SmallVector<int64_t> initThreadTile(initRank, 1);
      SmallVector<int64_t> initElementTile(initRank, 1);
      SmallVector<int64_t> initSubgroupStrides(initRank, 0);
      SmallVector<int64_t> initThreadStrides(initRank, 0);

      auto initLayout = IREE::VectorExt::NestedLayoutAttr::get(
          rewriter.getContext(), initSubgroupTile, initBatchTile, initOuterTile,
          initThreadTile, initElementTile, initSubgroupStrides, initThreadStrides);

      auto layoutedInitValue = IREE::VectorExt::ToLayoutOp::create(
          rewriter, loc, initValue, initLayout);
      auto layoutedInitIndex = IREE::VectorExt::ToLayoutOp::create(
          rewriter, loc, initIndex, initLayout);

      // Update init operands (positions depend on whether input_index is present)
      int initValueOperandIdx = argCompareOp.getInputIndex() ? 2 : 1;
      int initIndexOperandIdx = initValueOperandIdx + 1;

      argCompareOp->setOperand(initValueOperandIdx, layoutedInitValue.getResult());
      argCompareOp->setOperand(initIndexOperandIdx, layoutedInitIndex.getResult());
    }

    // Wrap results with the same layout as inits
    rewriter.setInsertionPointAfter(argCompareOp);
    if (initValueType.getRank() > 0) {
      SmallVector<int64_t> initShape(initValueType.getShape().begin(),
                                      initValueType.getShape().end());
      int64_t initRank = initShape.size();

      SmallVector<int64_t> resultSubgroupTile(initRank, 1);
      SmallVector<int64_t> resultBatchTile(initRank, 1);
      SmallVector<int64_t> resultOuterTile(initRank, 1);
      SmallVector<int64_t> resultThreadTile(initRank, 1);
      SmallVector<int64_t> resultElementTile(initRank, 1);
      SmallVector<int64_t> resultSubgroupStrides(initRank, 0);
      SmallVector<int64_t> resultThreadStrides(initRank, 0);

      auto resultLayout = IREE::VectorExt::NestedLayoutAttr::get(
          rewriter.getContext(), resultSubgroupTile, resultBatchTile, resultOuterTile,
          resultThreadTile, resultElementTile, resultSubgroupStrides, resultThreadStrides);

      auto layoutedResultValue = IREE::VectorExt::ToLayoutOp::create(
          rewriter, loc, argCompareOp.getResult(0), resultLayout);
      auto layoutedResultIndex = IREE::VectorExt::ToLayoutOp::create(
          rewriter, loc, argCompareOp.getResult(1), resultLayout);

      rewriter.replaceAllUsesExcept(argCompareOp.getResult(0),
                                    layoutedResultValue.getResult(),
                                    layoutedResultValue);
      rewriter.replaceAllUsesExcept(argCompareOp.getResult(1),
                                    layoutedResultIndex.getResult(),
                                    layoutedResultIndex);
    }
  }

  return success();
}

struct LLVMGPUVectorDistributePass final
    : impl::LLVMGPUVectorDistributePassBase<LLVMGPUVectorDistributePass> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<IREE::VectorExt::IREEVectorExtDialect>();
    registry.insert<affine::AffineDialect>();
    registry.insert<amdgpu::AMDGPUDialect>();
    registry.insert<gpu::GPUDialect>();
    registry.insert<scf::SCFDialect>();
  }

  void runOnOperation() override {
    mlir::FunctionOpInterface funcOp = getOperation();

    std::array<int64_t, 3> workgroupSize;
    if (funcOp->hasAttr("workgroup_size")) {
      auto tmpSizes =
          cast<ArrayAttr>(funcOp->getAttr("workgroup_size")).getValue();
      for (auto [i, size] : llvm::enumerate(tmpSizes)) {
        workgroupSize[i] = cast<IntegerAttr>(size).getInt();
      }
    } else {
      std::optional<SmallVector<int64_t>> maybeWorkgroupSize =
          getWorkgroupSize(funcOp);
      if (!maybeWorkgroupSize) {
        funcOp->emitOpError()
            << "unable to query workgroup_size information from entry point";
        return signalPassFailure();
      }
      for (auto [index, value] : llvm::enumerate(maybeWorkgroupSize.value())) {
        workgroupSize[index] = value;
      }
      for (auto index : llvm::seq<size_t>(maybeWorkgroupSize->size(), 3)) {
        workgroupSize[index] = 1;
      }
    }

    IRRewriter rewriter(funcOp);
    rewriter.setInsertionPointToStart(&funcOp.getFunctionBody().front());
    SmallVector<Value> threadGrid = {rewriter.createOrFold<gpu::ThreadIdOp>(
                                         funcOp.getLoc(), gpu::Dimension::z),
                                     rewriter.createOrFold<gpu::ThreadIdOp>(
                                         funcOp.getLoc(), gpu::Dimension::y),
                                     rewriter.createOrFold<gpu::ThreadIdOp>(
                                         funcOp.getLoc(), gpu::Dimension::x)};
    std::reverse(workgroupSize.begin(), workgroupSize.end());

    Value linearThreadIdVal = affine::AffineLinearizeIndexOp::create(
        rewriter, funcOp.getLoc(), threadGrid, workgroupSize,
        /*disjoint=*/true);

    std::optional<int64_t> subgroupSize = getSubgroupSize(funcOp);
    if (!subgroupSize) {
      funcOp->emitOpError()
          << "unable to query subgroup size information from entry point";
      return signalPassFailure();
    }

    // Note: arg_compare layout anchors are handled by layout analysis
    // The DistributeArgCompare pattern will handle distribution

    // Dump IR after setting anchors - DISABLED FOR NOW
    // fprintf(stderr, "\n========== IR After Setting Anchors ==========\n");
    // fflush(stderr);
    // funcOp->print(llvm::errs(), mlir::OpPrintingFlags().printGenericOpForm(false));
    // llvm::errs() << "\n";
    // llvm::errs().flush();
    // fprintf(stderr, "========== End IR Dump ==========\n\n");
    // fflush(stderr);

    ContractionVectorLayoutOptions options(funcOp, linearThreadIdVal,
                                           subgroupSize.value(), workgroupSize);

    LogicalResult result = distributeVectorOps(funcOp, options.getPatterns(), options);
    if (failed(result)) {
      funcOp->emitOpError() << "failed to distribute";
      return signalPassFailure();
    }
  }
};
} // namespace
} // namespace mlir::iree_compiler
