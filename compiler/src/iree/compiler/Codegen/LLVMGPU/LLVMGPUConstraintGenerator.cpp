// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/LLVMGPU/LLVMGPUConstraintGenerator.h"

#include "iree/compiler/Codegen/Common/GPU/GPUHeuristics.h"
#include "iree/compiler/Codegen/Common/SMTConstraintUtils.h"
#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenAttrs.h"
#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenInterfaces.h"
#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenOps.h"
#include "iree/compiler/Codegen/Dialect/GPU/IR/IREEGPUAttrs.h"
#include "iree/compiler/Codegen/Utils/GPUUtils.h"
#include "iree/compiler/Dialect/LinalgExt/IR/LinalgExtOps.h"
#include "llvm/ADT/StringExtras.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/SMT/IR/SMTOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/ValueRange.h"

namespace mlir::iree_compiler {

using AssertOp = IREE::Codegen::AssertOp;
using LookupOp = IREE::Codegen::LookupOp;
using IntKnobAttr = IREE::Codegen::IntKnobAttr;
using OneOfKnobAttr = IREE::Codegen::OneOfKnobAttr;
using RootOpAttr = IREE::Codegen::RootOpAttr;

namespace {

/// Contraction-like dimension classification used by both matmul and conv.
struct ContractionLikeDims {
  SmallVector<unsigned> b;
  SmallVector<unsigned> m;
  SmallVector<unsigned> n;
  SmallVector<unsigned> k;
};

/// Problem size, loop count, and indexing maps for a root op.
struct RootOpLoopInfo {
  SmallVector<int64_t> staticLoopRanges;
  unsigned numLoops;
  SmallVector<AffineMap> indexingMaps;
};

// Keys for entries in the SMT constraints `knobs` dictionary.
// These names are aligned with lowering config and translation info fields.
constexpr StringLiteral kKnobWorkgroupKey = "workgroup";
constexpr StringLiteral kKnobReductionKey = "reduction";
constexpr StringLiteral kKnobMmaKindKey = "mma_kind";
constexpr StringLiteral kKnobSubgroupBasisKey = "subgroup_basis";
constexpr StringLiteral kKnobSubgroupKey = "subgroup";
constexpr StringLiteral kKnobPromoteOperandsKey = "promote_operands";
constexpr StringLiteral kKnobWorkgroupSizeKey = "workgroup_size";
constexpr StringLiteral kKnobSubgroupSizeKey = "subgroup_size";

// SMT variable names for knob values used in constraints.
constexpr StringLiteral kKnobMmaIdxName = "mma_idx";
constexpr StringLiteral kKnobSgMCntName = "sg_m_cnt";
constexpr StringLiteral kKnobSgNCntName = "sg_n_cnt";
constexpr StringLiteral kKnobSgSizeName = "sg_size";
constexpr StringLiteral kKnobWgSizeXName = "wg_size_x";
constexpr StringLiteral kKnobWgSizeYName = "wg_size_y";
constexpr StringLiteral kKnobWgSizeZName = "wg_size_z";

// SMT variable name prefixes. The loop dim count varies
// per problem, so names are built at runtime as prefix + dim idx.
constexpr StringLiteral kKnobWgPrefix = "wg_";
constexpr StringLiteral kKnobRedPrefix = "red_";
// Per-loop subgroup tile size knobs (TileAndFuse). Sibling to `wg_`/`red_`:
// each one represents the number of MMA tiles a single subgroup covers along
// that loop. The subgroup *count* is derived (wg_tile / (sg_tile * mma)),
// not knobbed directly — different from VectorDistribute's sg_m_cnt/sg_n_cnt.
constexpr StringLiteral kKnobSgPrefix = "sg_";

// Default value for dims that are not knobbed.
constexpr int64_t kNoTileDimVal = 0;
constexpr int64_t kUnitTileDimVal = 1;

} // namespace

// TODO(#23535): These constraints are derived from existing tuner constraints.
// Need to verify the validity of the additional heuristic constraints,
// such as `red_k % mma_m == 0`.

/// Assert: lhs % rhs == 0, with format args for diagnostics.
static void assertDivisible(OpBuilder &builder, Location loc, Value lhs,
                            Value rhs, StringRef msg) {
  Value zero = mkIntConst(builder, loc, 0);
  Value rem = smt::IntModOp::create(builder, loc, lhs, rhs);
  Value eq = smt::EqOp::create(builder, loc, rem, zero);
  std::string fmtMsg = (msg + " ({} % {} == 0)").str();
  AssertOp::create(builder, loc, eq, fmtMsg, ValueRange{lhs, rhs});
}

/// Assert: lhs <pred> rhs
static void assertCmp(OpBuilder &builder, Location loc, smt::IntPredicate pred,
                      Value lhs, Value rhs, StringRef msg) {
  Value cmp = smt::IntCmpOp::create(builder, loc, pred, lhs, rhs);
  AssertOp::create(builder, loc, cmp, msg);
}

/// Assert: val >= lo && val <= hi.
static void assertBounds(OpBuilder &builder, Location loc, Value val,
                         StringRef name, Value lo, StringRef loName, Value hi,
                         StringRef hiName) {
  assertCmp(builder, loc, smt::IntPredicate::ge, val, lo,
            llvm::join_items("", name, " >= ", loName));
  assertCmp(builder, loc, smt::IntPredicate::le, val, hi,
            llvm::join_items("", name, " <= ", hiName));
}

/// Assert: lhs == rhs.
static void assertEq(OpBuilder &builder, Location loc, Value lhs, Value rhs,
                     StringRef msg) {
  Value eq = smt::EqOp::create(builder, loc, lhs, rhs);
  AssertOp::create(builder, loc, eq, msg);
}

/// Helper to build a knob variable name from a prefix.
/// e.g. ("wg_", 2) -> "wg_2".
static std::string makeVarName(StringRef prefix, unsigned idx) {
  return (prefix + Twine(idx)).str();
}

/// Helper to create an i64 IntegerAttr with a fixed value.
static IntegerAttr makeIntAttr(MLIRContext *ctx, int64_t value = 0) {
  return IntegerAttr::get(IntegerType::get(ctx, 64), value);
}

/// Get unique compatible MMA attrs for matmul and conv ops.
static SmallVector<Attribute>
getCompatibleMMAAttrs(linalg::LinalgOp op, IREE::GPU::TargetAttr gpuTarget,
                      const RootOpLoopInfo &loopInfo,
                      const ContractionLikeDims &dims) {
  if (gpuTarget.getWgp().getMma().empty()) {
    return {};
  }

  SmallVector<Attribute> mmaAttrs;
  const int64_t targetSubgroupSize = gpuTarget.getPreferredSubgroupSize();
  Type lhsElemType = getElementTypeOrSelf(op.getDpsInputOperand(0)->get());
  Type rhsElemType = getElementTypeOrSelf(op.getDpsInputOperand(1)->get());
  Type initElemType = getElementTypeOrSelf(op.getDpsInitOperand(0)->get());
  int64_t mSize = loopInfo.staticLoopRanges[dims.m.back()];
  int64_t nSize = loopInfo.staticLoopRanges[dims.n.back()];
  int64_t kSize = loopInfo.staticLoopRanges[dims.k.back()];

  // Dynamic shapes are not supported by tuner yet.
  if (ShapedType::isDynamic(mSize) || ShapedType::isDynamic(nSize) ||
      ShapedType::isDynamic(kSize)) {
    return {};
  }

  GPUMatmulShapeType problem{mSize,       nSize,       kSize,
                             lhsElemType, rhsElemType, initElemType};

  auto getIntrinsic = [](IREE::GPU::MMAAttr mma) -> GPUIntrinsicType {
    auto [mSize, nSize, kSize] = mma.getMNKShape();
    auto [aType, bType, cType] = mma.getABCElementTypes();
    return GPUIntrinsicType{mSize, nSize, kSize, aType, bType, cType, mma};
  };

  for (IREE::GPU::MMAAttr mma : gpuTarget.getWgp().getMma()) {
    if (mma.getSubgroupSize() != targetSubgroupSize) {
      continue;
    }
    // VectorDistribute matmul/conv skip block intrinsics.
    if (mma.isBlockIntrinsic()) {
      continue;
    }
    if (!mma.getDistributionMappingKind()) {
      continue;
    }
    // Check if the mma intrinsic supports the problem.
    if (failed(canTargetIntrinsic(problem, getIntrinsic(mma),
                                  targetSubgroupSize, /*canUpcastAcc*/ true,
                                  /*mustBeAligned*/ false))) {
      continue;
    }

    if (!llvm::is_contained(mmaAttrs, mma)) {
      mmaAttrs.push_back(mma);
    }
  }
  return mmaAttrs;
}

/// Get contraction-like (m,n,k) dims for a linalg op.
/// Only supports contraction and convolution today.
static FailureOr<ContractionLikeDims>
inferContractionLikeDims(linalg::LinalgOp linalgOp) {
  if (linalg::isaContractionOpInterface(linalgOp)) {
    FailureOr<mlir::linalg::ContractionDimensions> contractionDims =
        mlir::linalg::inferContractionDims(linalgOp);
    if (failed(contractionDims)) {
      return failure();
    }
    return ContractionLikeDims{llvm::to_vector(contractionDims->batch),
                               llvm::to_vector(contractionDims->m),
                               llvm::to_vector(contractionDims->n),
                               llvm::to_vector(contractionDims->k)};
  }
  if (linalg::isaConvolutionOpInterface(linalgOp)) {
    FailureOr<mlir::linalg::ConvolutionDimensions> convolutionDims =
        mlir::linalg::inferConvolutionDims(linalgOp);
    if (failed(convolutionDims) || convolutionDims->outputImage.empty() ||
        convolutionDims->outputChannel.empty() ||
        convolutionDims->inputChannel.empty()) {
      return failure();
    }
    // TODO(Amily): This mapping aligns with how VectorDistribute
    // sets the dims for convs. It may be too coarse for conv
    // semantics; revisit when plumbing through conv constraint
    // generation.
    return ContractionLikeDims{llvm::to_vector(convolutionDims->batch),
                               llvm::to_vector(convolutionDims->outputImage),
                               llvm::to_vector(convolutionDims->outputChannel),
                               llvm::to_vector(convolutionDims->inputChannel)};
  }
  return failure();
}

/// Returns loop info for supported root ops.
static std::optional<RootOpLoopInfo> getRootOpLoopInfo(Operation *rootOp) {
  if (auto linalgOp = dyn_cast<linalg::LinalgOp>(rootOp)) {
    return RootOpLoopInfo{linalgOp.getStaticLoopRanges(),
                          linalgOp.getNumLoops(),
                          linalgOp.getIndexingMapsArray()};
  }
  return std::nullopt;
}

/// Build the VectorDistribute knobs dict for contraction-like dims.
static DictionaryAttr
buildVectorDistributeKnobsDict(MLIRContext *ctx, const RootOpLoopInfo &loopInfo,
                               const ContractionLikeDims &dims,
                               ArrayRef<Attribute> compatibleMMAs) {
  SmallVector<NamedAttribute> knobsEntries;

  // Build workgroup entries from lowering config semantics: untiled dims get 0,
  // batch and outer M/N dims get unit tile 1, and inner M/N dims are knobbed.
  SmallVector<Attribute> workgroupEntries(loopInfo.numLoops,
                                          makeIntAttr(ctx, kNoTileDimVal));
  SmallVector<unsigned> unitWorkgroupDims;
  llvm::append_range(unitWorkgroupDims, dims.b);
  llvm::append_range(unitWorkgroupDims, dims.m);
  llvm::append_range(unitWorkgroupDims, dims.n);
  for (unsigned i : unitWorkgroupDims) {
    workgroupEntries[i] = makeIntAttr(ctx, kUnitTileDimVal);
  }
  // inferContractionLikeDims guarantees dims.m/n/k are non-empty for both
  // branches (asserted or early-returned for conv).
  workgroupEntries[dims.m.back()] =
      IntKnobAttr::get(ctx, makeVarName(kKnobWgPrefix, dims.m.back()));
  workgroupEntries[dims.n.back()] =
      IntKnobAttr::get(ctx, makeVarName(kKnobWgPrefix, dims.n.back()));
  knobsEntries.emplace_back(kKnobWorkgroupKey,
                            ArrayAttr::get(ctx, workgroupEntries));
  // Build reduction entries from the complement of unit workgroup dims.
  // Innermost K dim gets IntKnobAttr, outer K and filter dims get 1.
  SmallVector<Attribute> reductionEntries(loopInfo.numLoops,
                                          makeIntAttr(ctx, kNoTileDimVal));
  for (unsigned i = 0; i < loopInfo.numLoops; ++i) {
    if (llvm::is_contained(unitWorkgroupDims, i)) {
      continue;
    }
    reductionEntries[i] = makeIntAttr(ctx, kUnitTileDimVal);
  }
  reductionEntries[dims.k.back()] =
      IntKnobAttr::get(ctx, makeVarName(kKnobRedPrefix, dims.k.back()));
  knobsEntries.emplace_back(kKnobReductionKey,
                            ArrayAttr::get(ctx, reductionEntries));

  // Add mma_kind knob.
  knobsEntries.emplace_back(
      kKnobMmaKindKey,
      OneOfKnobAttr::get(ctx, kKnobMmaIdxName, compatibleMMAs));

  // Build subgroup_basis as [[counts], [mapping]]. Mapping is an identity map
  // of const int for VectorDistribute matmul and conv. It is kept as a
  // placeholder so downstream constraint verification can match the
  // subgroup_basis knob template in the lowering config.
  // Only innermost M and N dims get subgroup tiling, others stay 1.
  SmallVector<Attribute> subgroupCounts(loopInfo.numLoops, makeIntAttr(ctx, 1));
  subgroupCounts[dims.m.back()] = IntKnobAttr::get(ctx, kKnobSgMCntName);
  subgroupCounts[dims.n.back()] = IntKnobAttr::get(ctx, kKnobSgNCntName);
  SmallVector<Attribute> subgroupMapping;
  subgroupMapping.reserve(loopInfo.numLoops);
  for (unsigned i = 0; i < loopInfo.numLoops; ++i) {
    subgroupMapping.push_back(makeIntAttr(ctx, i));
  }
  ArrayAttr subgroupBasis =
      ArrayAttr::get(ctx, {ArrayAttr::get(ctx, subgroupCounts),
                           ArrayAttr::get(ctx, subgroupMapping)});
  knobsEntries.emplace_back(kKnobSubgroupBasisKey, subgroupBasis);

  // Add workgroup size and subgroup size at the top level.
  SmallVector<Attribute> wgSizeKnobs = {
      IntKnobAttr::get(ctx, kKnobWgSizeXName),
      IntKnobAttr::get(ctx, kKnobWgSizeYName),
      IntKnobAttr::get(ctx, kKnobWgSizeZName)};
  knobsEntries.emplace_back(kKnobWorkgroupSizeKey,
                            ArrayAttr::get(ctx, wgSizeKnobs));
  knobsEntries.emplace_back(kKnobSubgroupSizeKey,
                            IntKnobAttr::get(ctx, kKnobSgSizeName));

  return DictionaryAttr::get(ctx, knobsEntries);
}

/// Emit smt.lookup ops to derive MMA m/n/k shape values from the mma_idx knob.
/// Returns {mmaMLookup, mmaNLookup, mmaKLookup} SSA values.
/// Also asserts bounds on mma_idx: 0 <= mma_idx <= N-1.
static std::tuple<Value, Value, Value>
emitMMALookup(OpBuilder &builder, Location loc, ArrayRef<Attribute> mmaAttrs) {

  Value mmaIdx = mkKnob(builder, loc, kKnobMmaIdxName);
  Value minMmaIdxVal = mkIntConst(builder, loc, 0);
  int64_t maxMmaIdx = static_cast<int64_t>(mmaAttrs.size()) - 1;
  Value maxMmaIdxVal = mkIntConst(builder, loc, maxMmaIdx);
  std::string maxMmaIdxName = Twine(maxMmaIdx).str();
  // Assert bounds: 0 <= mma_idx <= N-1.
  assertBounds(builder, loc, mmaIdx, kKnobMmaIdxName, minMmaIdxVal, "0",
               maxMmaIdxVal, maxMmaIdxName);

  // Build lookup tables for m, n, k.
  SmallVector<int64_t> keys, mVals, nVals, kVals;
  for (auto [i, attr] : llvm::enumerate(mmaAttrs)) {
    auto [m, n, k] = cast<IREE::GPU::MmaInterfaceAttr>(attr).getMNKShape();
    keys.push_back(static_cast<int64_t>(i));
    mVals.push_back(m);
    nVals.push_back(n);
    kVals.push_back(k);
  }
  auto intTy = smt::IntType::get(builder.getContext());
  return {LookupOp::create(builder, loc, intTy, mmaIdx, keys, mVals),
          LookupOp::create(builder, loc, intTy, mmaIdx, keys, nVals),
          LookupOp::create(builder, loc, intTy, mmaIdx, keys, kVals)};
}

/// Emit VectorDistribute constraints for contraction-like dims (matmul/conv).
static LogicalResult emitVectorDistributeConstraints(
    OpBuilder &builder, linalg::LinalgOp linalgOp,
    const ContractionLikeDims &dims, IREE::GPU::TargetAttr gpuTarget,
    ArrayRef<Value> smtDimArgs, ArrayRef<Attribute> compatibleMMAs) {
  Location loc = linalgOp.getLoc();

  // Hardware constants.
  Value subgroupSizeVal =
      mkIntConst(builder, loc, gpuTarget.getPreferredSubgroupSize());
  Value maxThreadsVal = mkIntConst(
      builder, loc, gpuTarget.getWgp().getMaxThreadCountPerWorkgroup());
  Value maxSharedMemVal =
      mkIntConst(builder, loc, gpuTarget.getWgp().getMaxWorkgroupMemoryBytes());
  Value maxVGPRsVal = mkIntConst(builder, loc, 512);

  // Only innermost M, N, and K dims get constrained.
  unsigned mDim = dims.m.back();
  unsigned nDim = dims.n.back();
  unsigned kDim = dims.k.back();
  std::string wgMName = makeVarName(kKnobWgPrefix, mDim);
  std::string wgNName = makeVarName(kKnobWgPrefix, nDim);
  std::string redKName = makeVarName(kKnobRedPrefix, kDim);
  std::string mDimName = makeVarName(kLoopRangePrefix, mDim);
  std::string nDimName = makeVarName(kLoopRangePrefix, nDim);
  std::string kDimName = makeVarName(kLoopRangePrefix, kDim);

  // Create top-level knobs.
  Value wgM = mkKnob(builder, loc, wgMName);
  Value wgN = mkKnob(builder, loc, wgNName);
  Value redK = mkKnob(builder, loc, redKName);
  Value sgMCnt = mkKnob(builder, loc, kKnobSgMCntName);
  Value sgNCnt = mkKnob(builder, loc, kKnobSgNCntName);
  Value sgSize = mkKnob(builder, loc, kKnobSgSizeName);
  Value wgSizeX = mkKnob(builder, loc, kKnobWgSizeXName);
  Value wgSizeY = mkKnob(builder, loc, kKnobWgSizeYName);
  Value wgSizeZ = mkKnob(builder, loc, kKnobWgSizeZName);

  // Create intermediate variables.
  // Do not create these as knobs because they are not in the knob dict
  // and would fail constraint verification.
  // mma_m, mma_n, mma_k
  auto [mmaMLookup, mmaNLookup, mmaKLookup] =
      emitMMALookup(builder, loc, compatibleMMAs);
  // sg_num, number of subgroups per workgroup
  Value sgNum = smt::IntMulOp::create(builder, loc, ValueRange{sgMCnt, sgNCnt});
  Value sgMDenom =
      smt::IntMulOp::create(builder, loc, ValueRange{sgMCnt, mmaMLookup});
  // sg_m, number of mma_m along M per subgroup
  Value sgM = smt::IntDivOp::create(builder, loc, wgM, sgMDenom);
  Value sgNDenom =
      smt::IntMulOp::create(builder, loc, ValueRange{sgNCnt, mmaNLookup});
  // sg_n, number of mma_n along N per subgroup
  Value sgN = smt::IntDivOp::create(builder, loc, wgN, sgNDenom);
  // sg_k, number of mma_k along K per subgroup
  Value sgK = smt::IntDivOp::create(builder, loc, redK, mmaKLookup);

  // Constraint 0: Set concrete values for knobs.
  // sg_size == preferred_subgroup_size
  Value sgSizeEq = smt::EqOp::create(builder, loc, sgSize, subgroupSizeVal);
  AssertOp::create(builder, loc, sgSizeEq,
                   "sg_size == preferred_subgroup_size");

  // Constraint 1: Tile size must divide problem size.
  // dim % wg_tile == 0
  assertDivisible(
      builder, loc, smtDimArgs[mDim], wgM,
      llvm::join_items("", mDimName, " must be divisible by ", wgMName));
  assertDivisible(
      builder, loc, smtDimArgs[nDim], wgN,
      llvm::join_items("", nDimName, " must be divisible by ", wgNName));
  assertDivisible(
      builder, loc, smtDimArgs[kDim], redK,
      llvm::join_items("", kDimName, " must be divisible by ", redKName));

  // Constraint 2: Tile size bounds.
  // mma <= wg_tile <= dim
  assertBounds(builder, loc, wgM, wgMName, mmaMLookup, "mma_m",
               smtDimArgs[mDim], mDimName);
  assertBounds(builder, loc, wgN, wgNName, mmaNLookup, "mma_n",
               smtDimArgs[nDim], nDimName);
  assertBounds(builder, loc, redK, redKName, mmaKLookup, "mma_k",
               smtDimArgs[kDim], kDimName);
  // wg_tile <= 512 (max VGPRs)
  assertCmp(builder, loc, smt::IntPredicate::le, wgM, maxVGPRsVal,
            llvm::join_items("", wgMName, " <= 512 (max VGPRs)"));
  assertCmp(builder, loc, smt::IntPredicate::le, wgN, maxVGPRsVal,
            llvm::join_items("", wgNName, " <= 512 (max VGPRs)"));
  assertCmp(builder, loc, smt::IntPredicate::le, redK, maxVGPRsVal,
            llvm::join_items("", redKName, " <= 512 (max VGPRs)"));

  // Constraint 3: Tile size decomposition.
  // wg_tile % mma == 0
  assertDivisible(builder, loc, wgM, mmaMLookup,
                  llvm::join_items("", wgMName, " must be divisible by mma_m"));
  assertDivisible(builder, loc, wgN, mmaNLookup,
                  llvm::join_items("", wgNName, " must be divisible by mma_n"));
  assertDivisible(
      builder, loc, redK, mmaKLookup,
      llvm::join_items("", redKName, " must be divisible by mma_k"));
  // wg_m % (sg_m_cnt * mma_m * sg_m) == 0
  // wg_n % (sg_n_cnt * mma_n * sg_n) == 0
  // red_k % (1 * mma_k * sg_k) == 0
  Value mulResultM =
      smt::IntMulOp::create(builder, loc, ValueRange{sgMCnt, mmaMLookup, sgM});
  Value mulResultN =
      smt::IntMulOp::create(builder, loc, ValueRange{sgNCnt, mmaNLookup, sgN});
  Value mulResultK =
      smt::IntMulOp::create(builder, loc, ValueRange{mmaKLookup, sgK});
  assertDivisible(builder, loc, wgM, mulResultM,
                  llvm::join_items("", wgMName,
                                   " must be divisible by "
                                   "(sg_m_cnt * mma_m * sg_m)"));
  assertDivisible(builder, loc, wgN, mulResultN,
                  llvm::join_items("", wgNName,
                                   " must be divisible by "
                                   "(sg_n_cnt * mma_n * sg_n)"));

  assertDivisible(
      builder, loc, redK, mulResultK,
      llvm::join_items("", redKName, " must be divisible by (mma_k * sg_k)"));

  // Constraint 4: Subgroup count bounds.
  // wg_tile = mma * sg_tile * sg_cnt
  // Upper bound: max(wg_tile) / min(mma * 1) = 512 / 16 = 32.
  // 1 <= sg_m_cnt <= 32.
  // 1 <= sg_n_cnt <= 32.
  Value minSubgroupCntVal = mkIntConst(builder, loc, 1);
  Value maxSubgroupCntVal = mkIntConst(builder, loc, 32);
  assertBounds(builder, loc, sgMCnt, kKnobSgMCntName, minSubgroupCntVal, "1",
               maxSubgroupCntVal, "32");
  assertBounds(builder, loc, sgNCnt, kKnobSgNCntName, minSubgroupCntVal, "1",
               maxSubgroupCntVal, "32");
  // 1 <= sg_m <= 32.
  // 1 <= sg_n <= 32.
  // 1 <= sg_k <= 32.
  assertBounds(builder, loc, sgM, "sg_m", minSubgroupCntVal, "1",
               maxSubgroupCntVal, "32");
  assertBounds(builder, loc, sgN, "sg_n", minSubgroupCntVal, "1",
               maxSubgroupCntVal, "32");
  assertBounds(builder, loc, sgK, "sg_k", minSubgroupCntVal, "1",
               maxSubgroupCntVal, "32");
  // sg_m_cnt * sg_n_cnt == sg_num
  Value sgCntMulResult =
      smt::IntMulOp::create(builder, loc, ValueRange{sgMCnt, sgNCnt});
  Value sgCntEq = smt::EqOp::create(builder, loc, sgCntMulResult, sgNum);
  AssertOp::create(builder, loc, sgCntEq, "sg_m_cnt * sg_n_cnt == sg_num");
  // TODO(#23535): bounds range copied from current Tuner. Need to revisit.
  // 1 <= sg_num <= 10.
  assertBounds(builder, loc, sgNum, "sg_num", mkIntConst(builder, loc, 1), "1",
               mkIntConst(builder, loc, 10), "10");

  // Constraint 5: Thread count limit.
  // sg_m_cnt * sg_n_cnt * sg_size <= max_threads
  Value totalThreadsMulResult =
      smt::IntMulOp::create(builder, loc, ValueRange{sgMCnt, sgNCnt, sgSize});
  assertCmp(builder, loc, smt::IntPredicate::le, totalThreadsMulResult,
            maxThreadsVal, "total_threads <= max_threads");

  // Constraint 6: Load distribution.
  // (red_k * wg_m) % (sg_m_cnt * sg_n_cnt * sg_size) == 0
  // (red_k * wg_n) % (sg_m_cnt * sg_n_cnt * sg_size) == 0
  auto emitLoadDistForWg = [&](Value wg, StringRef name) -> void {
    Value lhsMulResult =
        smt::IntMulOp::create(builder, loc, ValueRange{redK, wg});
    Value rhsMulResult =
        smt::IntMulOp::create(builder, loc, ValueRange{sgMCnt, sgNCnt, sgSize});
    assertDivisible(
        builder, loc, lhsMulResult, rhsMulResult,
        llvm::join_items("", name, " must be divisible by thread_count"));
  };
  emitLoadDistForWg(wgM, "lhs_tile_elements");
  emitLoadDistForWg(wgN, "rhs_tile_elements");

  // Constraint 7: Shared memory limit.
  // Approximate formula:
  // (lhs_bytes * wg_m * red_k) + (rhs_bytes * wg_n * red_k)
  // Get element type bitwidths from operands.
  auto lhsType =
      cast<ShapedType>(linalgOp.getDpsInputOperand(0)->get().getType());
  auto rhsType =
      cast<ShapedType>(linalgOp.getDpsInputOperand(1)->get().getType());
  int64_t lhsBytes = lhsType.getElementTypeBitWidth() / 8;
  int64_t rhsBytes = rhsType.getElementTypeBitWidth() / 8;
  Value lhsBytesVal = mkIntConst(builder, loc, lhsBytes);
  Value rhsBytesVal = mkIntConst(builder, loc, rhsBytes);

  Value lhsMulResult =
      smt::IntMulOp::create(builder, loc, ValueRange{lhsBytesVal, wgM, redK});
  Value rhsMulResult =
      smt::IntMulOp::create(builder, loc, ValueRange{rhsBytesVal, wgN, redK});
  Value totalSharedMem = smt::IntAddOp::create(
      builder, loc, ValueRange{lhsMulResult, rhsMulResult});
  assertCmp(builder, loc, smt::IntPredicate::le, totalSharedMem,
            maxSharedMemVal, "shared memory must fit in workgroup memory");

  // Constraint 8: TranslationInfo workgroup structure.
  // For VectorDistribute, workgroup_size =
  // [wg_x, wg_y, wg_z] = [total_threads, 1, 1]
  // wg_x == total_threads
  Value wgXEq = smt::EqOp::create(builder, loc, wgSizeX, totalThreadsMulResult);
  AssertOp::create(builder, loc, wgXEq, "wg_size_x == total_threads");
  // wg_y == 1
  Value wgYEq =
      smt::EqOp::create(builder, loc, wgSizeY, mkIntConst(builder, loc, 1));
  AssertOp::create(builder, loc, wgYEq, "wg_size_y == 1");
  // wg_z == 1
  Value wgZEq =
      smt::EqOp::create(builder, loc, wgSizeZ, mkIntConst(builder, loc, 1));
  AssertOp::create(builder, loc, wgZEq, "wg_size_z == 1");

  // Constraint 9: Additional heuristic filters to narrow the search space.
  // wg_x <= wg_m OR wg_x <= wg_n
  Value wgXLeM =
      smt::IntCmpOp::create(builder, loc, smt::IntPredicate::le, wgSizeX, wgM);
  Value wgXLeN =
      smt::IntCmpOp::create(builder, loc, smt::IntPredicate::le, wgSizeX, wgN);
  Value orCond = smt::OrOp::create(builder, loc, ValueRange{wgXLeM, wgXLeN});
  AssertOp::create(builder, loc, orCond,
                   "wg_size_x <= wg_m OR wg_size_x <= wg_n");
  // red_k % mma_m == 0
  assertDivisible(
      builder, loc, redK, mmaMLookup,
      llvm::join_items("", redKName, " must be divisible by mma_m"));

  return success();
}

/// Emit constraints for a single root op under the VectorDistribute pipeline.
/// Only supports linalg contraction and convolution today.
static LogicalResult
emitVectorDistributeConstraintsForOp(Operation *rootOp, RootOpAttr rootOpAttr) {
  // Gate on contraction-like linalg ops.
  auto linalgOp = dyn_cast<linalg::LinalgOp>(rootOp);
  if (!linalgOp || (!linalg::isaContractionOpInterface(linalgOp) &&
                    !linalg::isaConvolutionOpInterface(linalgOp))) {
    return success();
  }

  IREE::GPU::TargetAttr gpuTarget = getGPUTargetAttr(rootOp);
  if (!gpuTarget) {
    return success();
  }

  std::optional<RootOpLoopInfo> loopInfo = getRootOpLoopInfo(rootOp);
  if (!loopInfo) {
    return success();
  }

  FailureOr<ContractionLikeDims> dims = inferContractionLikeDims(linalgOp);
  if (failed(dims)) {
    return success();
  }

  SmallVector<Attribute> compatibleMMAs =
      getCompatibleMMAAttrs(linalgOp, gpuTarget, *loopInfo, *dims);
  if (compatibleMMAs.empty()) {
    return success();
  }

  MLIRContext *ctx = rootOp->getContext();
  OpBuilder builder(ctx);
  DictionaryAttr knobs =
      buildVectorDistributeKnobsDict(ctx, *loopInfo, *dims, compatibleMMAs);
  auto pipelineAttr = IREE::GPU::PipelineAttr::get(
      ctx, IREE::GPU::LoweringPipeline::VectorDistribute);
  ConstraintsOpShell shell =
      createConstraintsOpShell(builder, rootOp, rootOpAttr, pipelineAttr, knobs,
                               loopInfo->numLoops, loopInfo->indexingMaps);

  return emitVectorDistributeConstraints(builder, linalgOp, *dims, gpuTarget,
                                         shell.smtDimArgs, compatibleMMAs);
}

/// Build the TileAndFuse knobs dict for a contraction (matmul) op.
///
/// Layout mirrors the TaF lowering_config schema observed in
/// `ROCDL/config_tile_and_fuse.mlir`:
///   workgroup        = [..., wg_<m>, wg_<n>, 0, ...]
///   reduction        = [..., 0, 0, ..., red_<k>]
///   subgroup         = [..., sg_<m>, sg_<n>, 0, ...]
///   mma_kind         = one_of_knob<mma_idx, [<compatible MMAs>]>
///   promote_operands = [0, 1]                    (fixed for v0)
///   workgroup_size   = [wg_size_x, 1, 1]         (TaF is 1D)
///   subgroup_size    = sg_size
///
/// Notes:
///   - Only innermost M/N/K loops carry knobs (v0); outer M/N/batch dims
///     are pinned to 1 and other loops to 0 — see PR #24484.
///   - workgroup_size y/z are fixed in the template rather than knobbed;
///     this avoids two trivial SMT assertions and lets the verifier reject
///     non-1D TaF lowering configs by template mismatch.
static DictionaryAttr
buildTileAndFuseKnobsDict(MLIRContext *ctx, const RootOpLoopInfo &loopInfo,
                          const ContractionLikeDims &dims,
                          ArrayRef<Attribute> compatibleMMAs) {
  SmallVector<NamedAttribute> knobsEntries;

  // workgroup: 0 for untiled dims, 1 for batch and outer M/N (only innermost
  // M/N are knobbed in v0).
  SmallVector<Attribute> workgroupEntries(loopInfo.numLoops,
                                          makeIntAttr(ctx, kNoTileDimVal));
  SmallVector<unsigned> unitWorkgroupDims;
  llvm::append_range(unitWorkgroupDims, dims.b);
  llvm::append_range(unitWorkgroupDims, dims.m);
  llvm::append_range(unitWorkgroupDims, dims.n);
  for (unsigned i : unitWorkgroupDims) {
    workgroupEntries[i] = makeIntAttr(ctx, kUnitTileDimVal);
  }
  workgroupEntries[dims.m.back()] =
      IntKnobAttr::get(ctx, makeVarName(kKnobWgPrefix, dims.m.back()));
  workgroupEntries[dims.n.back()] =
      IntKnobAttr::get(ctx, makeVarName(kKnobWgPrefix, dims.n.back()));
  knobsEntries.emplace_back(kKnobWorkgroupKey,
                            ArrayAttr::get(ctx, workgroupEntries));

  // reduction: complement of unit workgroup dims; innermost K is knobbed.
  SmallVector<Attribute> reductionEntries(loopInfo.numLoops,
                                          makeIntAttr(ctx, kNoTileDimVal));
  for (unsigned i = 0; i < loopInfo.numLoops; ++i) {
    if (llvm::is_contained(unitWorkgroupDims, i)) {
      continue;
    }
    reductionEntries[i] = makeIntAttr(ctx, kUnitTileDimVal);
  }
  reductionEntries[dims.k.back()] =
      IntKnobAttr::get(ctx, makeVarName(kKnobRedPrefix, dims.k.back()));
  knobsEntries.emplace_back(kKnobReductionKey,
                            ArrayAttr::get(ctx, reductionEntries));

  // subgroup: 0 elsewhere; innermost M/N carry per-loop sg_<dim> knobs.
  SmallVector<Attribute> subgroupEntries(loopInfo.numLoops,
                                         makeIntAttr(ctx, kNoTileDimVal));
  subgroupEntries[dims.m.back()] =
      IntKnobAttr::get(ctx, makeVarName(kKnobSgPrefix, dims.m.back()));
  subgroupEntries[dims.n.back()] =
      IntKnobAttr::get(ctx, makeVarName(kKnobSgPrefix, dims.n.back()));
  knobsEntries.emplace_back(kKnobSubgroupKey,
                            ArrayAttr::get(ctx, subgroupEntries));

  // mma_kind: discrete choice over compatible intrinsics.
  knobsEntries.emplace_back(
      kKnobMmaKindKey,
      OneOfKnobAttr::get(ctx, kKnobMmaIdxName, compatibleMMAs));

  // promote_operands: fixed [0, 1] for v0 matmul (lhs + rhs to LDS).
  SmallVector<Attribute> promoteEntries = {makeIntAttr(ctx, 0),
                                           makeIntAttr(ctx, 1)};
  knobsEntries.emplace_back(kKnobPromoteOperandsKey,
                            ArrayAttr::get(ctx, promoteEntries));

  // workgroup_size: TaF is 1D, so y and z are pinned to 1 in the template.
  SmallVector<Attribute> wgSizeKnobs = {IntKnobAttr::get(ctx, kKnobWgSizeXName),
                                        makeIntAttr(ctx, 1),
                                        makeIntAttr(ctx, 1)};
  knobsEntries.emplace_back(kKnobWorkgroupSizeKey,
                            ArrayAttr::get(ctx, wgSizeKnobs));
  knobsEntries.emplace_back(kKnobSubgroupSizeKey,
                            IntKnobAttr::get(ctx, kKnobSgSizeName));

  return DictionaryAttr::get(ctx, knobsEntries);
}

/// Emit TileAndFuse constraints for a matmul-like contraction.
/// Ported from the tuner's `generate_tile_and_fuse_constraints`
/// (amdsharktuner/rocm/rocm_dispatch_constraints.py lines 311-401), v0:
///   - innermost M/N/K dims only;
///   - per-loop subgroup tile sizes (TaF) instead of subgroup counts (VD);
///   - 1D workgroup;
///   - sub-byte element types currently rejected (TODO below).
///
/// TODO(#23535): The constants 512 (VGPR cap), 32 (subgroup-count cap), and
/// 10 (sg_num cap) come straight from the tuner; they should be derived
/// from the target descriptor once available.
static LogicalResult emitTileAndFuseConstraints(
    OpBuilder &builder, linalg::LinalgOp linalgOp,
    const ContractionLikeDims &dims, IREE::GPU::TargetAttr gpuTarget,
    ArrayRef<Value> smtDimArgs, ArrayRef<Attribute> compatibleMMAs) {
  Location loc = linalgOp.getLoc();

  // Sub-byte element types would silently zero out the shared-memory
  // budget below (`bitwidth / 8 == 0`). Skip emission for v0 — the
  // generic codegen still works, just without tuner pruning on this op.
  // TODO(#23535): switch to a bits-based LDS budget to support i4/fp4/i2.
  auto lhsType =
      cast<ShapedType>(linalgOp.getDpsInputOperand(0)->get().getType());
  auto rhsType =
      cast<ShapedType>(linalgOp.getDpsInputOperand(1)->get().getType());
  unsigned lhsBitWidth = lhsType.getElementTypeBitWidth();
  unsigned rhsBitWidth = rhsType.getElementTypeBitWidth();
  if (lhsBitWidth < 8 || rhsBitWidth < 8) {
    return success();
  }

  // Hardware constants.
  Value subgroupSizeVal =
      mkIntConst(builder, loc, gpuTarget.getPreferredSubgroupSize());
  Value maxThreadsVal = mkIntConst(
      builder, loc, gpuTarget.getWgp().getMaxThreadCountPerWorkgroup());
  Value maxSharedMemVal =
      mkIntConst(builder, loc, gpuTarget.getWgp().getMaxWorkgroupMemoryBytes());
  // VGPR-budgeted upper bound on a single tile axis (gfx942: 256 VGPRs/thread
  // at 64-thread subgroup with fp32 ≈ 512 elements/axis). Same magic number
  // as the VectorDistribute path; revisit once the target descriptor exposes
  // per-thread VGPR count.
  Value maxVGPRsVal = mkIntConst(builder, loc, 512);
  Value oneVal = mkIntConst(builder, loc, 1);

  // Only innermost M, N, and K dims get constrained (v0).
  unsigned mDim = dims.m.back();
  unsigned nDim = dims.n.back();
  unsigned kDim = dims.k.back();
  std::string wgMName = makeVarName(kKnobWgPrefix, mDim);
  std::string wgNName = makeVarName(kKnobWgPrefix, nDim);
  std::string redKName = makeVarName(kKnobRedPrefix, kDim);
  std::string sgMName = makeVarName(kKnobSgPrefix, mDim);
  std::string sgNName = makeVarName(kKnobSgPrefix, nDim);
  std::string mDimName = makeVarName(kLoopRangePrefix, mDim);
  std::string nDimName = makeVarName(kLoopRangePrefix, nDim);
  std::string kDimName = makeVarName(kLoopRangePrefix, kDim);

  // Top-level knobs.
  Value wgM = mkKnob(builder, loc, wgMName);
  Value wgN = mkKnob(builder, loc, wgNName);
  Value redK = mkKnob(builder, loc, redKName);
  Value sgM = mkKnob(builder, loc, sgMName);
  Value sgN = mkKnob(builder, loc, sgNName);
  Value sgSize = mkKnob(builder, loc, kKnobSgSizeName);
  Value wgSizeX = mkKnob(builder, loc, kKnobWgSizeXName);

  // Derived MMA shape values: mma_m, mma_n, mma_k indexed by the mma_idx
  // knob. Not in the knob dict, hence intentionally not declared via mkKnob.
  auto [mmaMLookup, mmaNLookup, mmaKLookup] =
      emitMMALookup(builder, loc, compatibleMMAs);

  // Derived subgroup counts. Unlike VectorDistribute, the subgroup *count*
  // along M/N is not a knob in TaF — it is recovered from the workgroup
  // tile, the subgroup tile, and the MMA shape:
  //   sg_m_cnt = wg_m / (sg_m * mma_m)
  //   sg_n_cnt = wg_n / (sg_n * mma_n)
  //   sg_num   = sg_m_cnt * sg_n_cnt
  // The divisions are guarded by Constraint 4 (divisibility) below.
  Value sgMDenom =
      smt::IntMulOp::create(builder, loc, ValueRange{sgM, mmaMLookup});
  Value sgNDenom =
      smt::IntMulOp::create(builder, loc, ValueRange{sgN, mmaNLookup});
  Value sgMCnt = smt::IntDivOp::create(builder, loc, wgM, sgMDenom);
  Value sgNCnt = smt::IntDivOp::create(builder, loc, wgN, sgNDenom);
  Value sgNum = smt::IntMulOp::create(builder, loc, ValueRange{sgMCnt, sgNCnt});
  // sg_k: number of mma_k along K per subgroup. Since red_k is already in
  // MMA-tile units (see element-space note above), sg_k == red_k.
  Value sgK = redK;

  // Constraint 0: sg_size pinned to the target's preferred subgroup size.
  assertEq(builder, loc, sgSize, subgroupSizeVal,
           "sg_size == preferred_subgroup_size");

  // Important unit note for the K axis (matches the Python tuner at
  // rocm_dispatch_constraints.py:343-358): `red_k` is in *MMA-tile units*,
  // not elements. The element-space K tile is `red_k * mma_k`. wg_m / wg_n
  // are in *elements* (lowering_config's workgroup field is element-sized
  // per the TaF pipeline; subgroup-tile knobs `sg_m`/`sg_n` are in MMA-tile
  // units, applied in constraints 4-7 below).
  Value redKElements =
      smt::IntMulOp::create(builder, loc, ValueRange{redK, mmaKLookup});

  // Constraint 1: problem size divisible by tile.
  assertDivisible(
      builder, loc, smtDimArgs[mDim], wgM,
      llvm::join_items("", mDimName, " must be divisible by ", wgMName));
  assertDivisible(
      builder, loc, smtDimArgs[nDim], wgN,
      llvm::join_items("", nDimName, " must be divisible by ", wgNName));
  assertDivisible(
      builder, loc, smtDimArgs[kDim], redKElements,
      llvm::join_items("", kDimName,
                       " must be divisible by (", redKName, " * mma_k)"));

  // Constraint 2: mma <= wg_tile <= dim (element-space for M/N), and
  // wg_tile <= VGPR budget. For K, the comparable check is on the
  // element-space size: mma_k <= red_k_elements <= dim_k.
  assertBounds(builder, loc, wgM, wgMName, mmaMLookup, "mma_m",
               smtDimArgs[mDim], mDimName);
  assertBounds(builder, loc, wgN, wgNName, mmaNLookup, "mma_n",
               smtDimArgs[nDim], nDimName);
  assertCmp(builder, loc, smt::IntPredicate::le, redKElements,
            smtDimArgs[kDim],
            llvm::join_items("", "(", redKName, " * mma_k) <= ", kDimName));
  // red_k itself is just a positive MMA-tile count.
  assertCmp(builder, loc, smt::IntPredicate::ge, redK, oneVal,
            llvm::join_items("", redKName, " >= 1"));
  assertCmp(builder, loc, smt::IntPredicate::le, wgM, maxVGPRsVal,
            llvm::join_items("", wgMName, " <= 512 (max VGPRs)"));
  assertCmp(builder, loc, smt::IntPredicate::le, wgN, maxVGPRsVal,
            llvm::join_items("", wgNName, " <= 512 (max VGPRs)"));
  assertCmp(builder, loc, smt::IntPredicate::le, redKElements, maxVGPRsVal,
            llvm::join_items("", "(", redKName,
                             " * mma_k) <= 512 (max VGPRs)"));

  // Constraint 3: wg_tile must be a whole multiple of the MMA shape so the
  // intrinsic tiles cleanly. K is already in MMA-tile units, so no
  // separate divisibility check is needed for it here.
  assertDivisible(builder, loc, wgM, mmaMLookup,
                  llvm::join_items("", wgMName, " must be divisible by mma_m"));
  assertDivisible(builder, loc, wgN, mmaNLookup,
                  llvm::join_items("", wgNName, " must be divisible by mma_n"));

  // Constraint 4: subgroup tile cleanly partitions the workgroup tile.
  //   wg_m % (sg_m * mma_m) == 0
  //   wg_n % (sg_n * mma_n) == 0
  // This is what makes the derived sg_m_cnt/sg_n_cnt exact integers.
  assertDivisible(
      builder, loc, wgM, sgMDenom,
      llvm::join_items("", wgMName, " must be divisible by (", sgMName,
                       " * mma_m)"));
  assertDivisible(
      builder, loc, wgN, sgNDenom,
      llvm::join_items("", wgNName, " must be divisible by (", sgNName,
                       " * mma_n)"));

  // Constraint 5: subgroup tiles are positive.
  assertCmp(builder, loc, smt::IntPredicate::ge, sgM, oneVal,
            llvm::join_items("", sgMName, " >= 1"));
  assertCmp(builder, loc, smt::IntPredicate::ge, sgN, oneVal,
            llvm::join_items("", sgNName, " >= 1"));

  // Constraint 6: derived subgroup counts in range.
  // TODO(#23535): 32 = max(wg_tile)/min(mma) = 512/16; revisit with the
  // VGPR-budget TODO above.
  Value maxSgCntVal = mkIntConst(builder, loc, 32);
  assertBounds(builder, loc, sgMCnt, "sg_m_cnt", oneVal, "1", maxSgCntVal,
               "32");
  assertBounds(builder, loc, sgNCnt, "sg_n_cnt", oneVal, "1", maxSgCntVal,
               "32");
  assertBounds(builder, loc, sgK, "sg_k", oneVal, "1", maxSgCntVal, "32");

  // Constraint 7: sg_num pinned to the tuner's default num_subgroups=4
  // (rocm_dispatch_constraints.py:387-389: when num_subgroups > 0 the
  // Python tuner emits `subgroups == num_subgroups`, not the loose
  // bounded form). Pinning here is what closes parity with the Python
  // counter; without it the compiler accepts ~4x more candidates.
  // TODO(#23535): expose num_subgroups as a search-space option (mirror
  // the `num_subgroups <= 0` Python branch with `1 <= sg_num <= 10`).
  Value numSubgroupsVal = mkIntConst(builder, loc, 4);
  assertEq(builder, loc, sgNum, numSubgroupsVal, "sg_num == 4");

  // Constraint 8: total thread count fits per-workgroup limit.
  Value totalThreads =
      smt::IntMulOp::create(builder, loc, ValueRange{sgNum, sgSize});
  assertCmp(builder, loc, smt::IntPredicate::le, totalThreads, maxThreadsVal,
            "total_threads <= max_threads");

  // Constraint 9: wg_size_x derivation. y and z are pinned to 1 by the
  // knob template, so only x needs an SMT-level equation.
  assertEq(builder, loc, wgSizeX, totalThreads,
           "wg_size_x == sg_num * sg_size");

  // Constraint 10: shared memory budget.
  //   bytes(lhs) * wg_m * red_k + bytes(rhs) * wg_n * red_k
  //     <= max_workgroup_memory_bytes
  // Approximate: ignores padding/double-buffering; matches the tuner's
  // formula at rocm_dispatch_constraints.py:394-399.
  int64_t lhsBytesPerElem = lhsBitWidth / 8;
  int64_t rhsBytesPerElem = rhsBitWidth / 8;
  Value lhsBytesVal = mkIntConst(builder, loc, lhsBytesPerElem);
  Value rhsBytesVal = mkIntConst(builder, loc, rhsBytesPerElem);
  // red_k is in MMA-tile units; use redKElements (= red_k * mma_k) for the
  // byte-accurate footprint.
  Value lhsSmemTerm =
      smt::IntMulOp::create(builder, loc, ValueRange{lhsBytesVal, wgM, redKElements});
  Value rhsSmemTerm =
      smt::IntMulOp::create(builder, loc, ValueRange{rhsBytesVal, wgN, redKElements});
  Value totalSmem =
      smt::IntAddOp::create(builder, loc, ValueRange{lhsSmemTerm, rhsSmemTerm});
  assertCmp(builder, loc, smt::IntPredicate::le, totalSmem, maxSharedMemVal,
            "shared memory must fit in workgroup memory");

  return success();
}

/// Emit constraints for a single root op under the TileAndFuse pipeline.
/// v0: matmul (contraction) only. Convolution-on-TaF is deferred (#23535).
static LogicalResult
emitTileAndFuseConstraintsForOp(Operation *rootOp, RootOpAttr rootOpAttr) {
  auto linalgOp = dyn_cast<linalg::LinalgOp>(rootOp);
  if (!linalgOp || !linalg::isaContractionOpInterface(linalgOp)) {
    return success();
  }

  IREE::GPU::TargetAttr gpuTarget = getGPUTargetAttr(rootOp);
  if (!gpuTarget) {
    return success();
  }

  std::optional<RootOpLoopInfo> loopInfo = getRootOpLoopInfo(rootOp);
  if (!loopInfo) {
    return success();
  }

  FailureOr<ContractionLikeDims> dims = inferContractionLikeDims(linalgOp);
  if (failed(dims)) {
    return success();
  }

  SmallVector<Attribute> compatibleMMAs =
      getCompatibleMMAAttrs(linalgOp, gpuTarget, *loopInfo, *dims);
  if (compatibleMMAs.empty()) {
    return success();
  }

  MLIRContext *ctx = rootOp->getContext();
  OpBuilder builder(ctx);
  DictionaryAttr knobs =
      buildTileAndFuseKnobsDict(ctx, *loopInfo, *dims, compatibleMMAs);
  auto pipelineAttr = IREE::GPU::PipelineAttr::get(
      ctx, IREE::GPU::LoweringPipeline::TileAndFuse);
  ConstraintsOpShell shell =
      createConstraintsOpShell(builder, rootOp, rootOpAttr, pipelineAttr, knobs,
                               loopInfo->numLoops, loopInfo->indexingMaps);

  return emitTileAndFuseConstraints(builder, linalgOp, *dims, gpuTarget,
                                    shell.smtDimArgs, compatibleMMAs);
}

/// Multiple root ops may be present in a set, e.g. <set = 0>:
/// [linalg.fill, linalg.matmul]. This function will choose the matmul op
/// over the fill op.
static Operation *getTunableOp(ArrayRef<Operation *> rootOps) {
  if (rootOps.empty()) {
    return nullptr;
  }
  for (Operation *rootOp : rootOps) {
    auto linalgOp = dyn_cast<linalg::LinalgOp>(rootOp);
    if (linalgOp && (linalg::isaContractionOpInterface(linalgOp) ||
                     linalg::isaConvolutionOpInterface(linalgOp))) {
      return rootOp;
    }
    if (auto attnOp = dyn_cast<IREE::LinalgExt::OnlineAttentionOp>(rootOp)) {
      return rootOp;
    }
  }
  return nullptr;
}

LogicalResult emitLLVMGPUConstraints(Attribute attr,
                                     ArrayRef<Operation *> rootOps) {
  Operation *tunableOp = getTunableOp(rootOps);
  if (!tunableOp) {
    return success();
  }
  RootOpAttr opAttr = getRootOpInfo(tunableOp);
  if (!opAttr) {
    return success();
  }

  auto gpuPipelineAttr = cast<IREE::GPU::PipelineAttr>(attr);

  switch (gpuPipelineAttr.getValue()) {
  case IREE::GPU::LoweringPipeline::VectorDistribute:
    return emitVectorDistributeConstraintsForOp(tunableOp, opAttr);
  case IREE::GPU::LoweringPipeline::TileAndFuse:
    return emitTileAndFuseConstraintsForOp(tunableOp, opAttr);
  default:
    // Other pipelines do not have constraint generation yet.
    return success();
  }
}

} // namespace mlir::iree_compiler
