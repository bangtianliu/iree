// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Common/GPU/GPUVectorDistribution.h"
#include "iree/compiler/Codegen/Common/Transforms.h"
#include "iree/compiler/Codegen/Dialect/VectorExt/IR/VectorExtDialect.h"
#include "iree/compiler/Codegen/Dialect/VectorExt/IR/VectorExtOps.h"
#include "iree/compiler/Codegen/Utils/GPUUtils.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Verifier.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Rewrite/PatternApplicator.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include <deque>

#define DEBUG_TYPE "iree-codegen-gpu-vector-distribution"

using namespace mlir::iree_compiler::IREE::VectorExt;

namespace mlir::iree_compiler {

using VectorValue = TypedValue<VectorType>;

constexpr StringLiteral kVectorLayoutFetcherStorageAttrName =
    "__vector_layout_fetcher_storage";

constexpr StringLiteral kVectorLayoutRedistributeAttrName =
    "__vector_layout_redistribute";

/// Set signature for the operation based on the analysis. Returns failure if
/// an operation contains vectors that cannot be distributed i.e. they have no
/// layout.
LogicalResult
setOpSignature(Operation *op,
               const llvm::MapVector<Value, VectorLayoutInterface> &layouts,
               const VectorLayoutOptions &options) {
  SmallVector<Attribute> operands;
  SmallVector<Attribute> results;

  // Count total operands including non-vectors and 0-D vectors for UnitAttr
  SmallVector<Attribute> allOperands(op->getNumOperands(), UnitAttr::get(op->getContext()));

  // Only fill in layouts for non-zero rank vectors
  for (auto [idx, operand] : llvm::enumerate(op->getOperands())) {
    if (auto vectorOperand = dyn_cast<VectorValue>(operand)) {
      // Check if this is a non-zero rank vector (needs distribution)
      if (isNonZeroRank(vectorOperand)) {
        if (auto layout = layouts.lookup(vectorOperand)) {
          allOperands[idx] = layout;
          continue;
        }
        if (auto layout = options.getDefaultLayout(vectorOperand.getType())) {
          allOperands[idx] = layout;
          continue;
        }
        return failure();
      }
      // 0-D vector (scalar) - already has UnitAttr from initialization
    }
    // Non-vector operands already have UnitAttr from initialization
  }
  operands = std::move(allOperands);

  // Count total results including non-vectors and 0-D vectors for UnitAttr
  SmallVector<Attribute> allResults(op->getNumResults(), UnitAttr::get(op->getContext()));

  // Only fill in layouts for non-zero rank vectors
  for (auto [idx, result] : llvm::enumerate(op->getResults())) {
    if (auto vectorResult = dyn_cast<VectorValue>(result)) {
      // Check if this is a non-zero rank vector (needs distribution)
      if (isNonZeroRank(vectorResult)) {
        if (auto layout = layouts.lookup(vectorResult)) {
          allResults[idx] = layout;
          continue;
        }
        if (auto layout = options.getDefaultLayout(vectorResult.getType())) {
          allResults[idx] = layout;
          continue;
        }
        return failure();
      }
      // 0-D vector (scalar) - already has UnitAttr from initialization
    }
    // Non-vector results already have UnitAttr from initialization
  }
  results = std::move(allResults);

  ArrayAttr operandsAttr = ArrayAttr::get(op->getContext(), operands);
  ArrayAttr resultsAttr = ArrayAttr::get(op->getContext(), results);
  Attribute signature[] = {operandsAttr, resultsAttr};
  op->setAttr(kVectorLayoutFetcherStorageAttrName,
              ArrayAttr::get(op->getContext(), signature));

  return success();
}

static bool hasOpSignature(Operation *op) {
  return op->hasAttrOfType<ArrayAttr>(kVectorLayoutFetcherStorageAttrName);
}

static DistributionSignature getOpSignature(Operation *op) {
  ArrayAttr signatureAttr =
      op->getAttrOfType<ArrayAttr>(kVectorLayoutFetcherStorageAttrName);
  assert(signatureAttr && "Op should have a signature attribute.");
  assert(signatureAttr.size() == 2 && "Malformed signature attribute.");

  ArrayAttr operandsAttr = dyn_cast<ArrayAttr>(signatureAttr[0]);
  ArrayAttr resultsAttr = dyn_cast<ArrayAttr>(signatureAttr[1]);
  assert(operandsAttr && resultsAttr && "Malformed signature attribute.");
  assert(operandsAttr.size() == op->getNumOperands() &&
         "Malformed signature attribute.");
  assert(resultsAttr.size() == op->getNumResults() &&
         "Malformed signature attribute.");

  DistributionSignature signature;

  auto addLayoutToSignature([&](Value value, Attribute layout) {
    // Unit attributes are used for non-vector values AND 0-D vectors (scalars).
    // Scalars are always distributed to all threads, so they don't get layouts.
    if (isa<UnitAttr>(layout)) {
      if (auto vectorValue = dyn_cast<VectorValue>(value)) {
        assert(!isNonZeroRank(vectorValue) &&
               "Malformed signature attribute: unit attribute for non-zero rank vector.");
      }
      return; // Don't add to signature map
    }

    assert(isa<VectorValue>(value) &&
           "Malformed signature attribute: non-unit attribute for non-vector "
           "value.");
    auto vector = cast<VectorValue>(value);

    auto vectorLayout = cast<VectorLayoutInterface>(layout);
    assert(vectorLayout && "Malformed signature attribute.");
    signature[vector] = vectorLayout;
  });

  for (auto [value, layout] :
       llvm::zip_equal(op->getOperands(), operandsAttr)) {
    addLayoutToSignature(value, layout);
  }
  for (auto [value, layout] : llvm::zip_equal(op->getResults(), resultsAttr)) {
    addLayoutToSignature(value, layout);
  }

  return signature;
}

VectorValue
DistributionPattern::getDistributed(RewriterBase &rewriter, VectorValue value,
                                    VectorLayoutInterface layout) const {
  // If this is a result of a "to_simd" op, use the source value of it.
  if (auto toSIMD = value.getDefiningOp<IREE::VectorExt::ToSIMDOp>()) {
    return cast<VectorValue>(toSIMD.getInput());
  }

  // Create a "to_simt" op to convert the value to the distributed layout.
  SmallVector<int64_t> distributedShape = layout.getDistributedShape();
  VectorType distributedType =
      VectorType::get(distributedShape, value.getType().getElementType());

  // For rank-0 vectors with empty distributed shape, the distributed type
  // is the same as the input type. Still create a ToSIMTOp for consistency.
  auto toSIMT = IREE::VectorExt::ToSIMTOp::create(rewriter, value.getLoc(),
                                                  distributedType, value);
  return toSIMT.getResult();
}

SmallVector<Value> DistributionPattern::getOpDistributedReplacements(
    RewriterBase &rewriter, Operation *op, ValueRange values) const {
  SmallVector<Value> replacements;
  for (auto [opResult, replacement] :
       llvm::zip_equal(op->getOpResults(), values)) {
    // If this value is a vector type, it must be converted back to simd.
    // However, 0-D vectors (scalars) don't need ToSIMD conversion.
    auto replacementVecType = dyn_cast<VectorType>(replacement.getType());
    auto originalVecType = dyn_cast<VectorType>(opResult.getType());

    // Only create to_simd if BOTH the replacement AND original are non-0D vectors
    if (replacementVecType && replacementVecType.getRank() != 0 &&
        originalVecType && originalVecType.getRank() != 0) {
      auto oldResult = cast<VectorValue>(opResult);
      // Create a toSIMD op to convert the value back to the simd.
      rewriter.setInsertionPointAfterValue(oldResult);
      Value toSIMD = IREE::VectorExt::ToSIMDOp::create(
          rewriter, oldResult.getLoc(), oldResult.getType(), replacement);
      // Add to replacements.
      replacement = toSIMD;
    } else if (replacementVecType && originalVecType &&
               replacementVecType != originalVecType) {
      // If types don't match (e.g., vector<1x1xi32> -> vector<i32>),
      // use shape_cast to convert
      rewriter.setInsertionPointAfterValue(opResult);
      replacement = vector::ShapeCastOp::create(
          rewriter, opResult.getLoc(), originalVecType, replacement);
    }
    replacements.push_back(replacement);
  }
  return replacements;
}

void DistributionPattern::replaceOpWithDistributedValues(
    RewriterBase &rewriter, Operation *op, ValueRange values) const {
  // Replace all OpResults with the given values.
  SmallVector<Value> replacements =
      getOpDistributedReplacements(rewriter, op, values);
  rewriter.replaceOp(op, replacements);
}

std::optional<DistributionSignature>
DistributionPattern::getOpSignature(Operation *op) const {
  if (!hasOpSignature(op)) {
    return std::nullopt;
  }
  return ::mlir::iree_compiler::getOpSignature(op);
}

void DistributionPattern::setSignatureForRedistribution(
    RewriterBase &rewriter, Operation *op,
    ArrayRef<VectorLayoutInterface> inputLayouts,
    ArrayRef<VectorLayoutInterface> outputLayouts) const {
  auto unitAttr = UnitAttr::get(rewriter.getContext());
  auto inputAttrs = SmallVector<Attribute>(op->getNumOperands(), unitAttr);
  auto outputAttrs = SmallVector<Attribute>(op->getNumResults(), unitAttr);

  auto isVectorType = [](Value x) { return isa<VectorType>(x.getType()); };

  // Only count non-0D vectors (0-D vectors are scalars and don't need layouts)
  // Use NDEBUG check to avoid unused variable warnings in Release builds
#ifndef NDEBUG
  auto isNon0DVectorType = [](Value x) {
    auto vecType = dyn_cast<VectorType>(x.getType());
    return vecType && vecType.getRank() > 0;
  };
  assert(llvm::count_if(op->getOperands(), isNon0DVectorType) ==
         static_cast<int64_t>(inputLayouts.size()));
#endif
  int64_t currVectorInput = 0;
  for (auto [idx, operand] : llvm::enumerate(op->getOperands())) {
    if (isVectorType(operand)) {
      // Skip 0-D vectors (scalars)
      if (cast<VectorType>(operand.getType()).getRank() == 0) {
        // Already initialized to unitAttr
        continue;
      }
      inputAttrs[idx] = inputLayouts[currVectorInput];
      ++currVectorInput;
    }
  }

  assert(llvm::count_if(op->getResults(), isNon0DVectorType) ==
         outputLayouts.size());
  int64_t currVectorOutput = 0;
  for (auto [idx, result] : llvm::enumerate(op->getResults())) {
    if (isVectorType(result)) {
      // Skip 0-D vectors (scalars)
      if (cast<VectorType>(result.getType()).getRank() == 0) {
        // Already initialized to unitAttr
        continue;
      }
      outputAttrs[idx] = outputLayouts[currVectorOutput];
      ++currVectorOutput;
    }
  }

  auto inputArrayAttr = ArrayAttr::get(rewriter.getContext(), inputAttrs);
  auto outputArrayAttr = ArrayAttr::get(rewriter.getContext(), outputAttrs);

  Attribute signature[] = {inputArrayAttr, outputArrayAttr};
  rewriter.modifyOpInPlace(op, [&]() {
    op->setAttr(kVectorLayoutFetcherStorageAttrName,
                ArrayAttr::get(rewriter.getContext(), signature));
    op->setAttr(kVectorLayoutRedistributeAttrName, unitAttr);
  });
}

LogicalResult
DistributionPattern::replaceParentMask(PatternRewriter &rewriter,
                                       vector::MaskOp maskOp) const {
  OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPoint(maskOp);
  std::optional<DistributionSignature> signatureMask = getOpSignature(maskOp);
  if (!signatureMask.has_value()) {
    return rewriter.notifyMatchFailure(maskOp, "mask should have a signature.");
  }
  SmallVector<Value> returns = maskOp.getBody()->getTerminator()->getOperands();
  for (auto [idx, ret] : llvm::enumerate(returns)) {
    if (VectorValue vectorRet = dyn_cast<VectorValue>(ret)) {
      VectorValue maskRet = cast<VectorValue>(maskOp.getResult(idx));
      VectorLayoutInterface layout =
          dyn_cast<NestedLayoutAttr>(signatureMask.value()[maskRet]);
      if (!layout) {
        return rewriter.notifyMatchFailure(maskOp,
                                           "layout must be NestedLayoutAttr");
      }
      ret = getDistributed(rewriter, vectorRet, layout);
    }
  }
  rewriter.eraseOp(maskOp.getBody()->getTerminator());
  rewriter.inlineBlockBefore(maskOp.getBody(), maskOp);
  replaceOpWithDistributedValues(rewriter, maskOp, returns);
  return success();
}

static void
debugPrintUniqueOperationNames(const std::deque<Operation *> &worklist) {
  DenseSet<StringRef> uniqueNames;
  for (Operation *op : worklist) {
    uniqueNames.insert(op->getName().getStringRef());
  }

  for (StringRef name : uniqueNames) {
    llvm::dbgs().indent(2) << "* " << name << "\n";
  }
  LLVM_DEBUG(llvm::dbgs() << "\n");
}

/// A rewriter for the pattern rewriting driver.
struct VectorDistributionRewriter : PatternRewriter {
  VectorDistributionRewriter(MLIRContext *ctx) : PatternRewriter(ctx) {}
};

/// Custom listener to store emitted ops that needs to be distributed.
struct VectorDistributionListener : public RewriterBase::Listener {
  bool hasOpsToBeDistributed() { return !toBeDistributed.empty(); }

  void clearOpsToBeDistributed() { return toBeDistributed.clear(); }

  const std::deque<Operation *> &getOpsToBeDistributed() const {
    return toBeDistributed;
  }

  void notifyOperationModified(Operation *op) override {
    if (op->hasAttr(kVectorLayoutRedistributeAttrName) &&
        op->hasAttrOfType<ArrayAttr>(kVectorLayoutFetcherStorageAttrName)) {
      op->removeAttr(kVectorLayoutRedistributeAttrName);
      toBeDistributed.push_back(op);
    }
  }

private:
  std::deque<Operation *> toBeDistributed;
};

static void applyVectorDistribution(Operation *root,
                                    const FrozenRewritePatternSet &patterns) {

  VectorDistributionRewriter rewriter(root->getContext());
  VectorDistributionListener listener;
  rewriter.setListener(&listener);
  PatternApplicator applicator(patterns);
  applicator.applyDefaultCostModel();

  // Collect all the operations to be distributed.
  std::deque<Operation *> worklist;
  LLVM_DEBUG(llvm::dbgs() << "Collecting operations to be distributed\n");
  root->walk([&](Operation *op) {
    // The distribution of mask op is special.
    // Although the signature set for visibility purposes
    // but it will be distributed when the body is
    // distributed. Therefore, we explicitly exclude
    // the yield and the mask op.
    if (hasOpSignature(op) && !isa<vector::MaskOp, vector::YieldOp>(op)) {
      worklist.push_back(op);
    }
  });
  LLVM_DEBUG(llvm::dbgs() << "Operations to be distributed:\n");
  LLVM_DEBUG(debugPrintUniqueOperationNames(worklist));

  // Note that the pattern application here never runs on a newly created
  // operation. It always runs on an existing operation. This ensures that no
  // invalidated state of the analysis is ever used.
  while (!worklist.empty()) {
    Operation *op = worklist.front();
    worklist.pop_front();
    if (op == nullptr) {
      continue;
    }

    LLVM_DEBUG(llvm::dbgs() << "Distributing: ");
    LLVM_DEBUG(op->print(llvm::dbgs(), OpPrintingFlags().skipRegions()));
    LLVM_DEBUG(llvm::dbgs() << "\n");

    if (failed(applicator.matchAndRewrite(op, rewriter))) {
      LLVM_DEBUG(llvm::dbgs().indent(2)
                 << ": Failed to distribute operation:\n");
      continue;
    }

    // Move recently emitted operations that needs to be distributed
    // from the local/rewriter worklist into the "global" worklist.
    if (listener.hasOpsToBeDistributed()) {
      auto opstoBeDistributed = listener.getOpsToBeDistributed();

      LLVM_DEBUG(llvm::dbgs()
                 << "Recently emitted operations to be distributed:\n");
      LLVM_DEBUG(debugPrintUniqueOperationNames(opstoBeDistributed));

      worklist.insert(worklist.end(), opstoBeDistributed.begin(),
                      opstoBeDistributed.end());
      listener.clearOpsToBeDistributed();
    }

    LLVM_DEBUG(llvm::dbgs().indent(2)
               << ": Successfully distributed operation:\n");
  }
}

LogicalResult distributeVectorOps(Operation *root,
                                  RewritePatternSet &distributionPatterns,
                                  VectorLayoutOptions &options) {
  // Run the analysis and determine the layouts.
  LLVM_DEBUG(llvm::dbgs() << "Running Layout Analysis\n");

  llvm::MapVector<Value, VectorLayoutInterface> layouts;
  if (failed(propagateVectorLayoutInfo(root, layouts))) {
    LLVM_DEBUG(llvm::dbgs() << "Layout Analysis Failed\n");
    return failure();
  }
  LLVM_DEBUG(llvm::dbgs() << "Layout Analysis Succeded\n");

  LLVM_DEBUG(llvm::dbgs() << "\n\n");

  // Go to each operation, and set its distribution signature.
  LLVM_DEBUG(
      llvm::dbgs() << "Setting distribution signatures for operations\n");
  root->walk([&](Operation *op) {
    if (failed(setOpSignature(op, layouts, options))) {
      LLVM_DEBUG({
        llvm::dbgs() << "Skipping operation because not all vector "
                        "operands/results have a layout:\n";
        op->print(llvm::dbgs());
      });
    }
  });
  LLVM_DEBUG(llvm::dbgs() << "Distribution signatures set\n");
  LLVM_DEBUG(root->print(llvm::dbgs()));
  LLVM_DEBUG(llvm::dbgs() << "\n\n");

  FrozenRewritePatternSet frozenPatterns(std::move(distributionPatterns));
  applyVectorDistribution(root, frozenPatterns);

  RewritePatternSet patterns(root->getContext());
  IREE::VectorExt::ToSIMDOp::getCanonicalizationPatterns(patterns,
                                                         root->getContext());
  IREE::VectorExt::ToSIMTOp::getCanonicalizationPatterns(patterns,
                                                         root->getContext());
  if (failed(applyPatternsGreedily(root, std::move(patterns)))) {
    return failure();
  }

  // Remove signature after distribution.
  root->walk([](Operation *op) {
    op->removeDiscardableAttr(kVectorLayoutFetcherStorageAttrName);
  });

  if (options.verifyConversion()) {
    WalkResult hasConversionOp = root->walk([](Operation *op) {
      if (isa<IREE::VectorExt::ToSIMDOp, IREE::VectorExt::ToSIMTOp>(op)) {
        for (auto user : op->getUsers()) {
          if (!isa<IREE::VectorExt::ToSIMDOp, IREE::VectorExt::ToSIMTOp>(
                  user)) {
            LLVM_DEBUG({
              llvm::dbgs() << "Found live cast op: " << *op << "\n";
              llvm::dbgs() << "With live user: " << *user << "\n";
            });
            return WalkResult::interrupt();
          }
        }
      }
      return WalkResult::advance();
    });
    if (hasConversionOp.wasInterrupted()) {
      return failure();
    }
  }
  return success();
}

} // namespace mlir::iree_compiler
