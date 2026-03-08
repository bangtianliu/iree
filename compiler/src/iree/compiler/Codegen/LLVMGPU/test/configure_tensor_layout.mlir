// RUN: iree-opt --split-input-file --pass-pipeline='builtin.module(func.func(iree-llvmgpu-configure-tensor-layouts, canonicalize, cse))' %s | FileCheck %s

#translation = #iree_codegen.translation_info<pipeline = LLVMGPUVectorDistribute
                                              workgroup_size = [64, 1, 1]
                                              subgroup_size = 64>

#maps = [
  affine_map<(m, n, k) -> (m, k)>,
  affine_map<(m, n, k) -> (n, k)>,
  affine_map<(m, n, k) -> (m, n)>
]

#traits = {
  indexing_maps = #maps,
  iterator_types = ["parallel", "parallel", "reduction"],
  lowering_config = #iree_gpu.lowering_config<{mma_kind = #iree_gpu.mma_layout<MFMA_F32_32x32x8_F16>,
                                              subgroup_basis = [[1, 1, 1], [0, 1, 2]]}>
}

func.func @matmul_96x64x16_mfma(%lhs: tensor<96x16xf16>,
                           %rhs: tensor<64x16xf16>,
                           %init: tensor<96x64xf32>)
                           -> tensor<96x64xf32>
                           attributes { translation_info = #translation } {
  %out = linalg.generic #traits
                        ins(%lhs, %rhs: tensor<96x16xf16>, tensor<64x16xf16>)
                        outs(%init: tensor<96x64xf32>) {
    ^bb0(%in: f16, %in_1: f16, %out: f32):
      %ex   = arith.extf %in   : f16 to f32
      %ex_1 = arith.extf %in_1 : f16 to f32
      %mul  = arith.mulf %ex, %ex_1 : f32
      %sum  = arith.addf %out, %mul : f32
      linalg.yield %sum : f32
  } -> tensor<96x64xf32>
  return %out : tensor<96x64xf32>
}

// CHECK-DAG: #[[$NESTED:.+]] = #iree_vector_ext.nested_layout<subgroup_tile = [1, 1], batch_tile = [3, 2], outer_tile = [1, 1], thread_tile = [32, 2], element_tile = [1, 4], subgroup_strides = [0, 0], thread_strides = [1, 32]>
// CHECK-DAG: #[[$NESTED1:.+]] = #iree_vector_ext.nested_layout<subgroup_tile = [1, 1], batch_tile = [2, 2], outer_tile = [1, 1], thread_tile = [32, 2], element_tile = [1, 4], subgroup_strides = [0, 0], thread_strides = [1, 32]>
// CHECK-DAG: #[[$NESTED2:.+]] = #iree_vector_ext.nested_layout<subgroup_tile = [1, 1], batch_tile = [3, 2], outer_tile = [4, 1], thread_tile = [2, 32], element_tile = [4, 1], subgroup_strides = [0, 0], thread_strides = [32, 1]>

// CHECK-LABEL: func.func @matmul_96x64x16_mfma

// CHECK-DAG: %[[LHS:.+]] = iree_vector_ext.to_layout %{{.*}} to layout(#[[$NESTED]])
// CHECK-DAG: %[[RHS:.+]] = iree_vector_ext.to_layout %{{.*}} to layout(#[[$NESTED1]])
// CHECK-DAG: %[[ACC:.+]] = iree_vector_ext.to_layout %{{.*}} to layout(#[[$NESTED2]])
// CHECK: linalg.generic
// CHECK-SAME: ins(%[[LHS]], %[[RHS]]
// CHECK-SAME: outs(%[[ACC]]

// -----

#translation = #iree_codegen.translation_info<pipeline = LLVMGPUVectorDistribute
                                              workgroup_size = [64, 1, 1]
                                              subgroup_size = 64>

#maps = [
  affine_map<(m, n, k) -> (m, k)>,
  affine_map<(m, n, k) -> (n, k)>,
  affine_map<(m, n, k) -> (m, n)>
]

#traits = {
  indexing_maps = #maps,
  iterator_types = ["parallel", "parallel", "reduction"],
  lowering_config = #iree_gpu.lowering_config<{mma_kind = #iree_gpu.mma_layout<WMMAR3_F32_16x16x16_F16>,
                                              subgroup_basis = [[1, 1, 1], [0, 1, 2]]}>
}

func.func @matmul_96x64x16_wmmar3(%lhs: tensor<96x16xf16>,
                           %rhs: tensor<64x16xf16>,
                           %init: tensor<96x64xf32>)
                           -> tensor<96x64xf32>
                           attributes { translation_info = #translation } {
  %out = linalg.generic #traits
                        ins(%lhs, %rhs: tensor<96x16xf16>, tensor<64x16xf16>)
                        outs(%init: tensor<96x64xf32>) {
    ^bb0(%in: f16, %in_1: f16, %out: f32):
      %ex   = arith.extf %in   : f16 to f32
      %ex_1 = arith.extf %in_1 : f16 to f32
      %mul  = arith.mulf %ex, %ex_1 : f32
      %sum  = arith.addf %out, %mul : f32
      linalg.yield %sum : f32
  } -> tensor<96x64xf32>
  return %out : tensor<96x64xf32>
}

// CHECK-DAG: #[[$NESTED:.+]] = #iree_vector_ext.nested_layout<subgroup_tile = [1, 1], batch_tile = [6, 1], outer_tile = [1, 1], thread_tile = [16, 1], element_tile = [1, 16], subgroup_strides = [0, 0], thread_strides = [1, 0]>
// CHECK-DAG: #[[$NESTED1:.+]] = #iree_vector_ext.nested_layout<subgroup_tile = [1, 1], batch_tile = [4, 1], outer_tile = [1, 1], thread_tile = [16, 1], element_tile = [1, 16], subgroup_strides = [0, 0], thread_strides = [1, 0]>
// CHECK-DAG: #[[$NESTED2:.+]] = #iree_vector_ext.nested_layout<subgroup_tile = [1, 1], batch_tile = [6, 4], outer_tile = [8, 1], thread_tile = [2, 16], element_tile = [1, 1], subgroup_strides = [0, 0], thread_strides = [16, 1]>

// CHECK-LABEL: func.func @matmul_96x64x16_wmmar3

// CHECK-DAG: %[[LHS:.+]] = iree_vector_ext.to_layout %{{.*}} to layout(#[[$NESTED]])
// CHECK-DAG: %[[RHS:.+]] = iree_vector_ext.to_layout %{{.*}} to layout(#[[$NESTED1]])
// CHECK-DAG: %[[ACC:.+]] = iree_vector_ext.to_layout %{{.*}} to layout(#[[$NESTED2]])
// CHECK: linalg.generic
// CHECK-SAME: ins(%[[LHS]], %[[RHS]]
// CHECK-SAME: outs(%[[ACC]]

// -----

#translation = #iree_codegen.translation_info<pipeline = LLVMGPUVectorDistribute
                                              workgroup_size = [64, 1, 1]
                                              subgroup_size = 64>

#maps = [
  affine_map<(m, n, k) -> (m, k)>,
  affine_map<(m, n, k) -> (n, k)>,
  affine_map<(m, n, k) -> (m, n)>
]

#traits = {
  indexing_maps = #maps,
  iterator_types = ["parallel", "parallel", "reduction"],
  lowering_config = #iree_gpu.lowering_config<{mma_kind = #iree_gpu.mma_layout<WMMAR4_F32_16x16x16_F16>,
                                              subgroup_basis = [[1, 1, 1], [0, 1, 2]]}>
}

func.func @matmul_96x64x16_wmmar4(%lhs: tensor<96x16xf16>,
                           %rhs: tensor<64x16xf16>,
                           %init: tensor<96x64xf32>)
                           -> tensor<96x64xf32>
                           attributes { translation_info = #translation } {
  %out = linalg.generic #traits
                        ins(%lhs, %rhs: tensor<96x16xf16>, tensor<64x16xf16>)
                        outs(%init: tensor<96x64xf32>) {
    ^bb0(%in: f16, %in_1: f16, %out: f32):
      %ex   = arith.extf %in   : f16 to f32
      %ex_1 = arith.extf %in_1 : f16 to f32
      %mul  = arith.mulf %ex, %ex_1 : f32
      %sum  = arith.addf %out, %mul : f32
      linalg.yield %sum : f32
  } -> tensor<96x64xf32>
  return %out : tensor<96x64xf32>
}

// CHECK-DAG: #[[$NESTED:.+]] = #iree_vector_ext.nested_layout<subgroup_tile = [1, 1], batch_tile = [6, 1], outer_tile = [1, 1], thread_tile = [16, 2], element_tile = [1, 8], subgroup_strides = [0, 0], thread_strides = [1, 16]>
// CHECK-DAG: #[[$NESTED1:.+]] = #iree_vector_ext.nested_layout<subgroup_tile = [1, 1], batch_tile = [4, 1], outer_tile = [1, 1], thread_tile = [16, 2], element_tile = [1, 8], subgroup_strides = [0, 0], thread_strides = [1, 16]>
// CHECK-DAG: #[[$NESTED2:.+]] = #iree_vector_ext.nested_layout<subgroup_tile = [1, 1], batch_tile = [6, 4], outer_tile = [1, 1], thread_tile = [2, 16], element_tile = [8, 1], subgroup_strides = [0, 0], thread_strides = [16, 1]>

// CHECK-LABEL: func.func @matmul_96x64x16_wmmar4

// CHECK-DAG: %[[LHS:.+]] = iree_vector_ext.to_layout %{{.*}} to layout(#[[$NESTED]])
// CHECK-DAG: %[[RHS:.+]] = iree_vector_ext.to_layout %{{.*}} to layout(#[[$NESTED1]])
// CHECK-DAG: %[[ACC:.+]] = iree_vector_ext.to_layout %{{.*}} to layout(#[[$NESTED2]])
// CHECK: linalg.generic
// CHECK-SAME: ins(%[[LHS]], %[[RHS]]
// CHECK-SAME: outs(%[[ACC]]

// -----

#translation = #iree_codegen.translation_info<pipeline = LLVMGPUVectorDistribute
                                              workgroup_size = [32, 1, 1]
                                              subgroup_size = 32>

#maps = [
  affine_map<(m, n, k) -> (m, k)>,
  affine_map<(m, n, k) -> (n, k)>,
  affine_map<(m, n, k) -> (m, n)>
]

#traits = {
  indexing_maps = #maps,
  iterator_types = ["parallel", "parallel", "reduction"],
  lowering_config = #iree_gpu.lowering_config<{mma_kind = #iree_gpu.mma_layout<WMMA_F32_16x16x32_F16>,
                                              subgroup_basis = [[1, 1, 1], [0, 1, 2]]}>
}

func.func @matmul_96x64x32_wmma_gfx1250(%lhs: tensor<96x32xf16>,
                                        %rhs: tensor<64x32xf16>,
                                        %init: tensor<96x64xf32>) -> tensor<96x64xf32>
                           attributes { translation_info = #translation } {
  %out = linalg.generic #traits
                        ins(%lhs, %rhs: tensor<96x32xf16>, tensor<64x32xf16>)
                        outs(%init: tensor<96x64xf32>) {
    ^bb0(%in: f16, %in_1: f16, %out: f32):
      %ex   = arith.extf %in   : f16 to f32
      %ex_1 = arith.extf %in_1 : f16 to f32
      %mul  = arith.mulf %ex, %ex_1 : f32
      %sum  = arith.addf %out, %mul : f32
      linalg.yield %sum : f32
  } -> tensor<96x64xf32>
  return %out : tensor<96x64xf32>
}

// CHECK-DAG: #[[$NESTED:.+]] = #iree_vector_ext.nested_layout<subgroup_tile = [1, 1], batch_tile = [6, 1], outer_tile = [1, 1], thread_tile = [16, 2], element_tile = [1, 16], subgroup_strides = [0, 0], thread_strides = [1, 16]>
// CHECK-DAG: #[[$NESTED1:.+]] = #iree_vector_ext.nested_layout<subgroup_tile = [1, 1], batch_tile = [4, 1], outer_tile = [1, 1], thread_tile = [16, 2], element_tile = [1, 16], subgroup_strides = [0, 0], thread_strides = [1, 16]>
// CHECK-DAG: #[[$NESTED2:.+]] = #iree_vector_ext.nested_layout<subgroup_tile = [1, 1], batch_tile = [6, 4], outer_tile = [1, 1], thread_tile = [2, 16], element_tile = [8, 1], subgroup_strides = [0, 0], thread_strides = [16, 1]>

// CHECK-LABEL: func.func @matmul_96x64x32_wmma_gfx1250

// CHECK-DAG: %[[LHS:.+]] = iree_vector_ext.to_layout %{{.*}} to layout(#[[$NESTED]])
// CHECK-DAG: %[[RHS:.+]] = iree_vector_ext.to_layout %{{.*}} to layout(#[[$NESTED1]])
// CHECK-DAG: %[[ACC:.+]] = iree_vector_ext.to_layout %{{.*}} to layout(#[[$NESTED2]])
// CHECK: linalg.generic
// CHECK-SAME: ins(%[[LHS]], %[[RHS]]
// CHECK-SAME: outs(%[[ACC]]

// -----

#translation = #iree_codegen.translation_info<pipeline = LLVMGPUVectorDistribute
                                              workgroup_size = [64, 1, 1]
                                              subgroup_size = 64>

#maps = [
  affine_map<(m, n, k) -> (m, k)>,
  affine_map<(m, n, k) -> (n, k)>,
  affine_map<(m, n, k) -> (m, n)>
]

#traits = {
  indexing_maps = #maps,
  iterator_types = ["parallel", "parallel", "reduction"],
  lowering_config = #iree_gpu.lowering_config<{mma_kind = #iree_gpu.mma_layout<MFMA_F32_16x16x16_F16>,
                                              subgroup_basis = [[4, 1, 1], [0, 1, 2]]}>
}

func.func @matmul_128x64x16_multi_subgroup(%lhs: tensor<128x16xf16>,
                                          %rhs: tensor<64x16xf16>,
                                          %init: tensor<128x64xf32>)
                                          -> tensor<128x64xf32>
                           attributes { translation_info = #translation } {
  %out = linalg.generic #traits
                        ins(%lhs, %rhs: tensor<128x16xf16>, tensor<64x16xf16>)
                        outs(%init: tensor<128x64xf32>) {
    ^bb0(%in: f16, %in_1: f16, %out: f32):
      %ex   = arith.extf %in   : f16 to f32
      %ex_1 = arith.extf %in_1 : f16 to f32
      %mul  = arith.mulf %ex, %ex_1 : f32
      %sum  = arith.addf %out, %mul : f32
      linalg.yield %sum : f32
  } -> tensor<128x64xf32>
  return %out : tensor<128x64xf32>
}

// CHECK-DAG: #[[$NESTED:.+]] = #iree_vector_ext.nested_layout<subgroup_tile = [4, 1]
// CHECK-DAG: #[[$NESTED1:.+]] = #iree_vector_ext.nested_layout<subgroup_tile = [1, 1]
// CHECK-DAG: #[[$NESTED2:.+]] = #iree_vector_ext.nested_layout<subgroup_tile = [4, 1]

// CHECK-LABEL: func.func @matmul_128x64x16_multi_subgroup

// CHECK-DAG: %[[LHS:.+]] = iree_vector_ext.to_layout %{{.*}} to layout(#[[$NESTED]])
// CHECK-DAG: %[[RHS:.+]] = iree_vector_ext.to_layout %{{.*}} to layout(#[[$NESTED1]])
// CHECK-DAG: %[[ACC:.+]] = iree_vector_ext.to_layout %{{.*}} to layout(#[[$NESTED2]])
// CHECK: linalg.generic
// CHECK-SAME: ins(%[[LHS]], %[[RHS]]
// CHECK-SAME: outs(%[[ACC]]

// -----

#translation = #iree_codegen.translation_info<pipeline = LLVMGPUVectorDistribute
                                              workgroup_size = [64, 1, 1]
                                              subgroup_size = 64>

func.func @linalg_copy(%in : tensor<16x16x16xf16>) -> tensor<16x16x16xf16>
                      attributes { translation_info = #translation } {
  %empty = tensor.empty() : tensor<16x16x16xf16>
  %copied = linalg.copy
            { lowering_config = #iree_gpu.derived_thread_config }
            ins(%in : tensor<16x16x16xf16>)
            outs(%empty : tensor<16x16x16xf16>) -> tensor<16x16x16xf16>
  func.return %copied : tensor<16x16x16xf16>
}

// CHECK-DAG: #[[$LAYOUT:.+]] = #iree_vector_ext.nested_layout<subgroup_tile = [1, 1, 1], batch_tile = [8, 1, 1], outer_tile = [1, 1, 1], thread_tile = [2, 16, 2], element_tile = [1, 1, 8], subgroup_strides = [0, 0, 0], thread_strides = [32, 2, 1]>

// CHECK-LABEL: func.func @linalg_copy
// CHECK: %[[OUT:.+]] = linalg.copy
// CHECK: to_layout %[[OUT]] to layout(#[[$LAYOUT]])

// -----

#translation = #iree_codegen.translation_info<pipeline = LLVMGPUVectorDistribute
                                              workgroup_size = [64, 1, 1]
                                              subgroup_size = 64>

#map = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2)>
#map1 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d3, d4, d5)>

#gather_trait = {
    indexing_maps = [affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2)>,
                     affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d3, d4, d5)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel", "parallel"],
    lowering_config = #iree_gpu.derived_thread_config
}

func.func @gather_like(%base : tensor<16384x16x32x128xf16>,
                       %indices : tensor<4x64x4xi64>)
                       -> tensor<4x64x4x16x32x128xf16>
                       attributes { translation_info = #translation } {

  %empty = tensor.empty() : tensor<4x64x4x16x32x128xf16>
  %gather = linalg.generic #gather_trait
            ins(%indices : tensor<4x64x4xi64>)
            outs(%empty : tensor<4x64x4x16x32x128xf16>) {
  ^bb0(%in: i64, %out: f16):
    %idx = arith.index_cast %in : i64 to index
    %iv3 = linalg.index 3 : index
    %iv4 = linalg.index 4 : index
    %iv5 = linalg.index 5 : index
    %extracted = tensor.extract %base[%idx, %iv3, %iv4, %iv5] : tensor<16384x16x32x128xf16>
    linalg.yield %extracted : f16
  } -> tensor<4x64x4x16x32x128xf16>

  func.return %gather : tensor<4x64x4x16x32x128xf16>
}

// CHECK-DAG: #[[$LAYOUT:.+]] = #iree_vector_ext.nested_layout<subgroup_tile = [1, 1, 1, 1, 1, 1], batch_tile = [4, 64, 4, 16, 8, 1], outer_tile = [1, 1, 1, 1, 1, 1], thread_tile = [1, 1, 1, 1, 4, 16], element_tile = [1, 1, 1, 1, 1, 8], subgroup_strides = [0, 0, 0, 0, 0, 0], thread_strides = [0, 0, 0, 0, 16, 1]>

// CHECK-LABEL: func.func @gather_like
// CHECK: %[[OUT:.+]] = linalg.generic
// CHECK: to_layout %[[OUT]] to layout(#[[$LAYOUT]])

// -----

#translation = #iree_codegen.translation_info<pipeline = LLVMGPUVectorDistribute
                                              workgroup_size = [64, 1, 1]
                                              subgroup_size = 64>

func.func @dynamic_infer_sizes(%in : tensor<4x32x?x128xf16>) -> tensor<1x1x?x128xf16> attributes { translation_info = #translation } {
  %c0 = arith.constant 0 : index
  %c2 = arith.constant 2 : index
  %d2 = tensor.dim %in, %c2 : tensor<4x32x?x128xf16>
  %45 = affine.min affine_map<(d0)[s0] -> (-d0 + s0, 1024)>(%c0)[%d2]
  %extracted_slice_5 = tensor.extract_slice %in[%c0, %c0, %c0, 0] [1, 1, %45, 128] [1, 1, 1, 1] : tensor<4x32x?x128xf16> to tensor<1x1x?x128xf16>
  %49 = tensor.empty(%45) : tensor<1x1x?x128xf16>
  %50 = linalg.copy {lowering_config = #iree_gpu.derived_thread_config} ins(%extracted_slice_5 : tensor<1x1x?x128xf16>) outs(%49 : tensor<1x1x?x128xf16>) -> tensor<1x1x?x128xf16>
  return %50 : tensor<1x1x?x128xf16>
}

// CHECK-DAG: #[[LAYOUT:.+]] = #iree_vector_ext.nested_layout<subgroup_tile = [1, 1, 1, 1], batch_tile = [1, 1, 256, 1], outer_tile = [1, 1, 1, 1], thread_tile = [1, 1, 4, 16], element_tile = [1, 1, 1, 8], subgroup_strides = [0, 0, 0, 0], thread_strides = [0, 0, 16, 1]>

// CHECK: %[[EXTRACT:.+]] = tensor.extract_slice %arg0{{.*}} : tensor<4x32x?x128xf16> to tensor<1x1x?x128xf16>
// CHECK: %[[EMPTY:.+]] = tensor.empty({{.*}}) : tensor<1x1x?x128xf16>
// CHECK: %[[COPY:.+]] = linalg.copy {{.*}} ins(%[[EXTRACT]] : tensor<1x1x?x128xf16>) outs(%[[EMPTY]] : tensor<1x1x?x128xf16>)
// CHECK: iree_vector_ext.to_layout %[[COPY]] to layout(#[[LAYOUT]]) : tensor<1x1x?x128xf16>

// -----

#translation = #iree_codegen.translation_info<pipeline = LLVMGPUVectorDistribute
                                              workgroup_size = [64, 1, 1]
                                              subgroup_size = 64>

#lowering_config = #iree_gpu.lowering_config<{
    subgroup_basis = [[1, 1, 2, 2], [0, 1, 2, 3]],
    lane_basis = [[1, 1, 8, 8], [0, 1, 2, 3]],
    thread = [0, 0, 8, 8]
}>

func.func @dynamic_infer_sizes_lowering_config(%in : tensor<4x32x?x128xf16>) -> tensor<1x1x?x128xf16> attributes { translation_info = #translation } {
  %c0 = arith.constant 0 : index
  %c2 = arith.constant 2 : index
  %d2 = tensor.dim %in, %c2 : tensor<4x32x?x128xf16>
  %45 = affine.min affine_map<(d0)[s0] -> (-d0 + s0, 128)>(%c0)[%d2]
  %extracted_slice_5 = tensor.extract_slice %in[%c0, %c0, %c0, 0] [1, 1, %45, 128] [1, 1, 1, 1] : tensor<4x32x?x128xf16> to tensor<1x1x?x128xf16>
  %49 = tensor.empty(%45) : tensor<1x1x?x128xf16>
  %50 = linalg.copy {lowering_config = #lowering_config} ins(%extracted_slice_5 : tensor<1x1x?x128xf16>) outs(%49 : tensor<1x1x?x128xf16>) -> tensor<1x1x?x128xf16>
  return %50 : tensor<1x1x?x128xf16>
}

// CHECK-DAG: #[[LAYOUT:.+]] = #iree_vector_ext.nested_layout<subgroup_tile = [1, 1, 2, 2], batch_tile = [1, 1, 1, 1], outer_tile = [1, 1, 1, 1], thread_tile = [1, 1, 8, 8], element_tile = [1, 1, 8, 8], subgroup_strides = [0, 0, 2, 1], thread_strides = [0, 0, 8, 1]>

// CHECK: %[[EXTRACT:.+]] = tensor.extract_slice %arg0{{.*}} : tensor<4x32x?x128xf16> to tensor<1x1x?x128xf16>
// CHECK: %[[EMPTY:.+]] = tensor.empty({{.*}}) : tensor<1x1x?x128xf16>
// CHECK: %[[EXTRACTL:.+]] = iree_vector_ext.to_layout %[[EXTRACT]] to layout(#[[LAYOUT]]) : tensor<1x1x?x128xf16>
// CHECK: %[[EMPTYL:.+]] = iree_vector_ext.to_layout %[[EMPTY]] to layout(#[[LAYOUT]]) : tensor<1x1x?x128xf16>
// CHECK: %[[COPY:.+]] = linalg.copy {{.*}} ins(%[[EXTRACTL]] : tensor<1x1x?x128xf16>) outs(%[[EMPTYL]] : tensor<1x1x?x128xf16>)
// CHECK: iree_vector_ext.to_layout %[[COPY]] to layout(#[[LAYOUT]]) : tensor<1x1x?x128xf16>

// -----

// Verify that the batch tile for a dimension that requires ceil division
// (63 / 8 = 8, not 7) is computed correctly.

#translation = #iree_codegen.translation_info<pipeline = LLVMGPUVectorDistribute
                                              workgroup_size = [512, 1, 1]
                                              subgroup_size = 64>

#maps = [
  affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>,
  affine_map<(d0, d1, d2, d3) -> (d0, d3, d2)>,
  affine_map<(d0, d1, d2, d3) -> (d0, d3, d1)>
]

#traits = {
  indexing_maps = #maps,
  iterator_types = ["parallel", "parallel", "reduction", "parallel"],
  lowering_config = #iree_gpu.lowering_config<{
    lane_basis = [[1, 1, 1, 1, 64], [1, 0, 3, 4]],
    subgroup_basis = [[1, 1, 1, 1, 8], [0, 1, 2, 4]],
    thread = [0, 0, 8, 0]
  }>
}

func.func @contraction_ceildiv_batch(%lhs: tensor<1x1x63xf16>,
                                     %rhs: tensor<1x512x63xf16>,
                                     %init: tensor<1x512x1xf32>)
                                     -> tensor<1x512x1xf32>
                                     attributes { translation_info = #translation } {
  %out = linalg.generic #traits
                        ins(%lhs, %rhs: tensor<1x1x63xf16>, tensor<1x512x63xf16>)
                        outs(%init: tensor<1x512x1xf32>) {
    ^bb0(%in: f16, %in_1: f16, %out: f32):
      %ex   = arith.extf %in   : f16 to f32
      %ex_1 = arith.extf %in_1 : f16 to f32
      %mul  = arith.mulf %ex, %ex_1 : f32
      %sum  = arith.addf %mul, %out : f32
      linalg.yield %sum : f32
  } -> tensor<1x512x1xf32>
  return %out : tensor<1x512x1xf32>
}

// CHECK-DAG: #[[$NESTED:.+]] = #iree_vector_ext.nested_layout<{{.*}}batch_tile = [1, 1, 8]{{.*}}element_tile = [1, 1, 8]{{.*}}>
// CHECK-DAG: #[[$NESTED1:.+]] = #iree_vector_ext.nested_layout<{{.*}}batch_tile = [1, 1, 8]{{.*}}element_tile = [1, 1, 8]{{.*}}>

// CHECK-LABEL: func.func @contraction_ceildiv_batch

// CHECK-DAG: %[[LHS:.+]] = iree_vector_ext.to_layout %{{.*}} to layout(#[[$NESTED]])
// CHECK-DAG: %[[RHS:.+]] = iree_vector_ext.to_layout %{{.*}} to layout(#[[$NESTED1]])
// CHECK: linalg.generic
// CHECK-SAME: ins(%[[LHS]], %[[RHS]]

// -----

#translation_propagate = #iree_codegen.translation_info<pipeline = LLVMGPUVectorDistribute
                                              workgroup_size = [64, 1, 1]
                                              subgroup_size = 64>

func.func @argcompare_layout_propagation(%input: tensor<16x16x16xf16>) -> (tensor<16x16xf16>, tensor<16x16xi32>)
    attributes { translation_info = #translation_propagate } {
  %cst = arith.constant 1.0 : f16
  %init_val = tensor.empty() : tensor<16x16xf16>
  %init_idx = tensor.empty() : tensor<16x16xi32>
  %init_processed = tensor.empty() : tensor<16x16x16xf16>

  // Producer op that sets layouts via lowering_config.
  %processed = linalg.generic {
      lowering_config = #iree_gpu.derived_thread_config,
      indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>,
                       affine_map<(d0, d1, d2) -> (d0, d1, d2)>],
      iterator_types = ["parallel", "parallel", "parallel"]}
      ins(%input : tensor<16x16x16xf16>)
      outs(%init_processed : tensor<16x16x16xf16>) {
    ^bb0(%in: f16, %out: f16):
      %add = arith.addf %in, %cst : f16
      linalg.yield %add : f16
  } -> tensor<16x16x16xf16>

  // ArgCompare propagates layouts from its producer.
  %result:2 = iree_linalg_ext.arg_compare
      dimension(2)
      ins(%processed : tensor<16x16x16xf16>)
      outs(%init_val, %init_idx : tensor<16x16xf16>, tensor<16x16xi32>) {
    ^bb0(%lhs: f16, %rhs: f16):
      %cmp = arith.cmpf ogt, %lhs, %rhs : f16
      iree_linalg_ext.yield %cmp : i1
  } -> tensor<16x16xf16>, tensor<16x16xi32>
  return %result#0, %result#1 : tensor<16x16xf16>, tensor<16x16xi32>
}

// Verify layouts flow from producer to arg_compare input.
// CHECK-LABEL: func.func @argcompare_layout_propagation
// CHECK: %[[GENERIC:.+]] = linalg.generic
// CHECK: %[[LAYOUT_GENERIC:.+]] = iree_vector_ext.to_layout %[[GENERIC]] to layout(#{{.+}})
// CHECK: iree_linalg_ext.arg_compare
// CHECK-SAME: ins(%[[LAYOUT_GENERIC]] : tensor<16x16x16xf16>)

// -----

// Test layout propagation through scf.for loop results to arg_compare.

#translation_scf = #iree_codegen.translation_info<pipeline = LLVMGPUVectorDistribute
                                              workgroup_size = [64, 1, 1]
                                              subgroup_size = 64>

func.func @argcompare_scf_for_propagation(%input: tensor<16x16x16xf16>)
    -> (tensor<16x16xf16>, tensor<16x16xi32>)
    attributes { translation_info = #translation_scf } {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %cst = arith.constant 1.0 : f16
  %init_val = tensor.empty() : tensor<16x16xf16>
  %init_idx = tensor.empty() : tensor<16x16xi32>

  %loop_result = scf.for %iv = %c0 to %c4 step %c1
      iter_args(%arg = %input) -> tensor<16x16x16xf16> {
    %init_processed = tensor.empty() : tensor<16x16x16xf16>
    %processed = linalg.generic {
        lowering_config = #iree_gpu.derived_thread_config,
        indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>,
                         affine_map<(d0, d1, d2) -> (d0, d1, d2)>],
        iterator_types = ["parallel", "parallel", "parallel"]}
        ins(%arg : tensor<16x16x16xf16>)
        outs(%init_processed : tensor<16x16x16xf16>) {
      ^bb0(%in: f16, %out: f16):
        %add = arith.addf %in, %cst : f16
        linalg.yield %add : f16
    } -> tensor<16x16x16xf16>
    scf.yield %processed : tensor<16x16x16xf16>
  }

  %result:2 = iree_linalg_ext.arg_compare
      dimension(2)
      ins(%loop_result : tensor<16x16x16xf16>)
      outs(%init_val, %init_idx : tensor<16x16xf16>, tensor<16x16xi32>) {
    ^bb0(%lhs: f16, %rhs: f16):
      %cmp = arith.cmpf ogt, %lhs, %rhs : f16
      iree_linalg_ext.yield %cmp : i1
  } -> tensor<16x16xf16>, tensor<16x16xi32>
  return %result#0, %result#1 : tensor<16x16xf16>, tensor<16x16xi32>
}

// CHECK-LABEL: func.func @argcompare_scf_for_propagation
// CHECK: scf.for
// CHECK:   %[[GENERIC:.+]] = linalg.generic
// CHECK:   %[[LAYOUT_INSIDE:.+]] = iree_vector_ext.to_layout %[[GENERIC]] to layout(#{{.+}})
// CHECK:   scf.yield %[[LAYOUT_INSIDE]]
// CHECK: }
// CHECK: %[[LAYOUT_PROPAGATED:.+]] = iree_vector_ext.to_layout %{{.+}} to layout(#{{.+}})
// CHECK: iree_linalg_ext.arg_compare
// CHECK-SAME: ins(%[[LAYOUT_PROPAGATED]] : tensor<16x16x16xf16>)

// -----

// Test arg_compare with explicit lowering_config.

#translation_with_config = #iree_codegen.translation_info<pipeline = LLVMGPUVectorDistribute
                                              workgroup_size = [64, 1, 1]
                                              subgroup_size = 64>

#lowering_config_argcompare = #iree_gpu.lowering_config<{
    workgroup = [1, 1, 0],
    partial_reduction = [0, 0, 16],
    thread = [1, 1, 4],
    lane_basis = [[1, 1, 4], [0, 1, 2]],
    subgroup_basis = [[1, 1, 1], [0, 1, 2]]
}>

func.func @argcompare_with_lowering_config(%input: tensor<16x16x16xf16>)
    -> (tensor<16x16xf16>, tensor<16x16xi32>)
    attributes { translation_info = #translation_with_config } {
  %init_val = tensor.empty() : tensor<16x16xf16>
  %init_idx = tensor.empty() : tensor<16x16xi32>

  %result:2 = iree_linalg_ext.arg_compare {
      lowering_config = #lowering_config_argcompare
    }
      dimension(2)
      ins(%input : tensor<16x16x16xf16>)
      outs(%init_val, %init_idx : tensor<16x16xf16>, tensor<16x16xi32>) {
    ^bb0(%lhs: f16, %rhs: f16):
      %cmp = arith.cmpf ogt, %lhs, %rhs : f16
      iree_linalg_ext.yield %cmp : i1
  } -> tensor<16x16xf16>, tensor<16x16xi32>

  return %result#0, %result#1 : tensor<16x16xf16>, tensor<16x16xi32>
}

// Verify layouts are applied to all operands and results.
// CHECK-LABEL: func.func @argcompare_with_lowering_config
// CHECK: %[[EMPTY0:.+]] = tensor.empty() : tensor<16x16xf16>
// CHECK: %[[EMPTY1:.+]] = tensor.empty() : tensor<16x16xi32>
// CHECK: %[[IN_LAYOUT:.+]] = iree_vector_ext.to_layout %{{.+}} to layout(#{{.+}}) : tensor<16x16x16xf16>
// CHECK: %[[OUT0_LAYOUT:.+]] = iree_vector_ext.to_layout %[[EMPTY0]] to layout(#{{.+}}) : tensor<16x16xf16>
// CHECK: %[[OUT1_LAYOUT:.+]] = iree_vector_ext.to_layout %[[EMPTY1]] to layout(#{{.+}}) : tensor<16x16xi32>
// CHECK: %[[RESULT:.+]]:2 = iree_linalg_ext.arg_compare
// CHECK-SAME: lowering_config
// CHECK-SAME: dimension(2)
// CHECK-SAME: ins(%[[IN_LAYOUT]] : tensor<16x16x16xf16>)
// CHECK-SAME: outs(%[[OUT0_LAYOUT]], %[[OUT1_LAYOUT]] : tensor<16x16xf16>, tensor<16x16xi32>)
// CHECK: iree_vector_ext.to_layout %[[RESULT]]#0
// CHECK: iree_vector_ext.to_layout %[[RESULT]]#1

// -----

// Test that multiple arg_compare operations can coexist and each receives
// the correct propagated layout without interference.

#translation_multi = #iree_codegen.translation_info<pipeline = LLVMGPUVectorDistribute
                                              workgroup_size = [64, 1, 1]
                                              subgroup_size = 64>

func.func @multiple_argcompare_ops(%input: tensor<16x16x16xf16>)
    -> (tensor<16x16xf16>, tensor<16x16xi32>, tensor<16x16xf16>, tensor<16x16xi32>)
    attributes { translation_info = #translation_multi } {
  %cst = arith.constant 1.0 : f16
  %init_val1 = tensor.empty() : tensor<16x16xf16>
  %init_idx1 = tensor.empty() : tensor<16x16xi32>
  %init_val2 = tensor.empty() : tensor<16x16xf16>
  %init_idx2 = tensor.empty() : tensor<16x16xi32>
  %init_processed = tensor.empty() : tensor<16x16x16xf16>

  // Generic op with lowering_config that will set layouts.
  %processed = linalg.generic {
      lowering_config = #iree_gpu.derived_thread_config,
      indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>,
                       affine_map<(d0, d1, d2) -> (d0, d1, d2)>],
      iterator_types = ["parallel", "parallel", "parallel"]}
      ins(%input : tensor<16x16x16xf16>)
      outs(%init_processed : tensor<16x16x16xf16>) {
    ^bb0(%in: f16, %out: f16):
      %add = arith.addf %in, %cst : f16
      linalg.yield %add : f16
  } -> tensor<16x16x16xf16>

  // First arg_compare (max along dimension 2)
  %max_result:2 = iree_linalg_ext.arg_compare
      dimension(2)
      ins(%processed : tensor<16x16x16xf16>)
      outs(%init_val1, %init_idx1 : tensor<16x16xf16>, tensor<16x16xi32>) {
    ^bb0(%lhs: f16, %rhs: f16):
      %cmp = arith.cmpf ogt, %lhs, %rhs : f16
      iree_linalg_ext.yield %cmp : i1
  } -> tensor<16x16xf16>, tensor<16x16xi32>

  // Second arg_compare (min along same dimension, using same input)
  %min_result:2 = iree_linalg_ext.arg_compare
      dimension(2)
      ins(%processed : tensor<16x16x16xf16>)
      outs(%init_val2, %init_idx2 : tensor<16x16xf16>, tensor<16x16xi32>) {
    ^bb0(%lhs: f16, %rhs: f16):
      %cmp = arith.cmpf olt, %lhs, %rhs : f16
      iree_linalg_ext.yield %cmp : i1
  } -> tensor<16x16xf16>, tensor<16x16xi32>

  return %max_result#0, %max_result#1, %min_result#0, %min_result#1 : tensor<16x16xf16>, tensor<16x16xi32>, tensor<16x16xf16>, tensor<16x16xi32>
}

// Both arg_compare operations should receive the propagated layout from the same linalg.generic.
// CHECK-LABEL: func.func @multiple_argcompare_ops
// CHECK: %[[GENERIC:.+]] = linalg.generic
// CHECK: %[[LAYOUT_GENERIC:.+]] = iree_vector_ext.to_layout %[[GENERIC]] to layout(#{{.+}})
// First arg_compare gets the layout.
// CHECK: iree_linalg_ext.arg_compare
// CHECK-SAME: ins(%[[LAYOUT_GENERIC]] : tensor<16x16x16xf16>)
// Second arg_compare also gets the same layout.
// CHECK: iree_linalg_ext.arg_compare
// CHECK-SAME: ins(%[[LAYOUT_GENERIC]] : tensor<16x16x16xf16>)
