// RUN: iree-opt --split-input-file \
// RUN:   --iree-codegen-experimental-verify-pipeline-constraints \
// RUN:   --pass-pipeline='builtin.module(func.func(iree-codegen-insert-smt-constraints))' %s \
// RUN:   | FileCheck %s

// Per-constraint coverage for the TileAndFuse emitter. The dispatcher in
// emitLLVMGPUConstraints also runs the VectorDistribute emitter for the
// same dispatch (both pipelines are registered on the AMD target), but
// these CHECK lines are written to ignore the VD constraints op and lock
// in the TaF op's knob template and assert families. The companion test
// config_llvm_gpu_constraints.mlir locks in the VD specifics.

#gpu_target = #iree_gpu.target<arch = "gfx942", features = "", wgp = <
  compute = fp32, storage = b32, subgroup = shuffle,
  mma = [<MFMA_F32_16x16x16_F16>],
  subgroup_size_choices = [64],
  max_load_instruction_bits = 128,
  max_workgroup_sizes = [1024, 1024, 1024],
  max_thread_count_per_workgroup = 1024,
  max_workgroup_memory_bytes = 65536,
  max_workgroup_counts = [2147483647, 2147483647, 2147483647]
>>
#exec_target = #hal.executable.target<"rocm", "rocm-hsaco-fb",
    {iree_codegen.target_info = #gpu_target}>

//
// Test 1: Plain matmul. Locks in the TaF knob template (workgroup,
// reduction, subgroup, mma_kind, promote_operands, workgroup_size,
// subgroup_size) and verifies all 14 v0 assert families.
//

func.func @matmul() attributes {hal.executable.target = #exec_target} {
  %lhs = tensor.empty() : tensor<128x64xf16>
  %rhs = tensor.empty() : tensor<64x256xf16>
  %empty = tensor.empty() : tensor<128x256xf32>
  %result = linalg.matmul {root_op = #iree_codegen.root_op<set = 0>}
      ins(%lhs, %rhs : tensor<128x64xf16>, tensor<64x256xf16>)
      outs(%empty : tensor<128x256xf32>) -> tensor<128x256xf32>
  return
}

// CHECK-LABEL: func.func @matmul
// CHECK:       linalg.matmul {{.+}} #iree_codegen.root_op<set = [[SET:[0-9]+]]>
// CHECK:       iree_codegen.smt.constraints target = <set = [[SET]]>, pipeline = #iree_gpu.pipeline<TileAndFuse>,
// CHECK-NEXT:  knobs = {
// CHECK-DAG:   mma_kind = #iree_codegen.smt.one_of_knob<"mma_idx", [#iree_gpu.mma_layout<MFMA_F32_16x16x16_F16>]>
// CHECK-DAG:   promote_operands = [0, 1]
// CHECK-DAG:   reduction = [0, 0, #iree_codegen.smt.int_knob<"red_2">]
// CHECK-DAG:   subgroup = [#iree_codegen.smt.int_knob<"sg_0">, #iree_codegen.smt.int_knob<"sg_1">, 0]
// CHECK-DAG:   subgroup_size = #iree_codegen.smt.int_knob<"sg_size">
// CHECK-DAG:   workgroup = [#iree_codegen.smt.int_knob<"wg_0">, #iree_codegen.smt.int_knob<"wg_1">, 0]
// CHECK-DAG:   workgroup_size = [#iree_codegen.smt.int_knob<"wg_size_x">, 1, 1]
// CHECK-SAME:  }

// Constraint families. CHECK-DAG so the order inside the region doesn't
// matter; one per assert message family that the emitter must produce.
// CHECK-DAG: "sg_size == preferred_subgroup_size"
// CHECK-DAG: "dim_0 must be divisible by wg_0 ({} % {} == 0)"
// CHECK-DAG: "dim_1 must be divisible by wg_1 ({} % {} == 0)"
// CHECK-DAG: "dim_2 must be divisible by (red_2 * mma_k) ({} % {} == 0)"
// CHECK-DAG: "wg_0 >= mma_m"
// CHECK-DAG: "wg_0 <= dim_0"
// CHECK-DAG: "wg_1 >= mma_n"
// CHECK-DAG: "wg_1 <= dim_1"
// CHECK-DAG: "(red_2 * mma_k) <= dim_2"
// CHECK-DAG: "red_2 >= 1"
// CHECK-DAG: "wg_0 <= 512 (max VGPRs)"
// CHECK-DAG: "wg_1 <= 512 (max VGPRs)"
// CHECK-DAG: "(red_2 * mma_k) <= 512 (max VGPRs)"
// CHECK-DAG: "wg_0 must be divisible by mma_m ({} % {} == 0)"
// CHECK-DAG: "wg_1 must be divisible by mma_n ({} % {} == 0)"
// CHECK-DAG: "wg_0 must be divisible by (sg_0 * mma_m) ({} % {} == 0)"
// CHECK-DAG: "wg_1 must be divisible by (sg_1 * mma_n) ({} % {} == 0)"
// CHECK-DAG: "sg_0 >= 1"
// CHECK-DAG: "sg_1 >= 1"
// CHECK-DAG: "sg_m_cnt >= 1"
// CHECK-DAG: "sg_m_cnt <= 32"
// CHECK-DAG: "sg_n_cnt >= 1"
// CHECK-DAG: "sg_n_cnt <= 32"
// CHECK-DAG: "sg_k >= 1"
// CHECK-DAG: "sg_k <= 32"
// CHECK-DAG: "sg_num == 4"
// CHECK-DAG: "total_threads <= max_threads"
// CHECK-DAG: "wg_size_x == sg_num * sg_size"
// CHECK-DAG: "shared memory must fit in workgroup memory"

// -----

//
// Test 2: Expanded (batched) matmul through linalg.generic.
// Verifies the knob template stays consistent with multi-loop ops: outer
// batch + outer M/N dims get `1`s in workgroup, knobs only on innermost
// M/N/K.
//

#gpu_target_2 = #iree_gpu.target<arch = "gfx942", features = "", wgp = <
  compute = fp32, storage = b32, subgroup = shuffle,
  mma = [<MFMA_F32_16x16x16_F16>],
  subgroup_size_choices = [64],
  max_load_instruction_bits = 128,
  max_workgroup_sizes = [1024, 1024, 1024],
  max_thread_count_per_workgroup = 1024,
  max_workgroup_memory_bytes = 65536,
  max_workgroup_counts = [2147483647, 2147483647, 2147483647]
>>
#exec_target_2 = #hal.executable.target<"rocm", "rocm-hsaco-fb",
    {iree_codegen.target_info = #gpu_target_2}>

#map_lhs = affine_map<(d0, d1, d2, d3, d4) -> (d0, d2, d4)>
#map_rhs = affine_map<(d0, d1, d2, d3, d4) -> (d1, d3, d4)>
#map_out = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3)>

func.func @expanded_matmul()
    attributes {hal.executable.target = #exec_target_2} {
  %lhs = tensor.empty() : tensor<2x64x2048xf16>
  %rhs = tensor.empty() : tensor<10x64x2048xf16>
  %empty = tensor.empty() : tensor<2x10x64x64xf32>
  %result = linalg.generic {
      indexing_maps = [#map_lhs, #map_rhs, #map_out],
      iterator_types = ["parallel", "parallel", "parallel", "parallel",
                        "reduction"],
      root_op = #iree_codegen.root_op<set = 0>}
      ins(%lhs, %rhs : tensor<2x64x2048xf16>,
                       tensor<10x64x2048xf16>)
      outs(%empty : tensor<2x10x64x64xf32>) {
  ^bb0(%in_lhs: f16, %in_rhs: f16, %out: f32):
    %ext_l = arith.extf %in_lhs : f16 to f32
    %ext_r = arith.extf %in_rhs : f16 to f32
    %mul = arith.mulf %ext_l, %ext_r : f32
    %add = arith.addf %mul, %out : f32
    linalg.yield %add : f32
  } -> tensor<2x10x64x64xf32>
  return
}

// CHECK-LABEL: func.func @expanded_matmul
// CHECK:       linalg.generic
// CHECK:       iree_codegen.smt.constraints target = <set = 0>, pipeline = #iree_gpu.pipeline<TileAndFuse>,
// CHECK-NEXT:  knobs = {
// Outer M/N dims get unit tile (1); innermost M/N (loop indices 2/3) carry knobs.
// CHECK-DAG:   workgroup = [1, 1, #iree_codegen.smt.int_knob<"wg_2">, #iree_codegen.smt.int_knob<"wg_3">, 0]
// CHECK-DAG:   subgroup = [0, 0, #iree_codegen.smt.int_knob<"sg_2">, #iree_codegen.smt.int_knob<"sg_3">, 0]
// CHECK-DAG:   reduction = [0, 0, 0, 0, #iree_codegen.smt.int_knob<"red_4">]
// CHECK-DAG:   promote_operands = [0, 1]
// CHECK-DAG:   workgroup_size = [#iree_codegen.smt.int_knob<"wg_size_x">, 1, 1]
// CHECK-SAME:  }
// CHECK-DAG: "dim_2 must be divisible by wg_2 ({} % {} == 0)"
// CHECK-DAG: "dim_3 must be divisible by wg_3 ({} % {} == 0)"
// CHECK-DAG: "dim_4 must be divisible by (red_4 * mma_k) ({} % {} == 0)"

// -----

//
// Test 3: No emission when no MMA matches (Type/element compatibility).
// All-int target → contraction with f16 inputs has no compatible MMA.
// Neither the TaF nor the VD emitter should produce a constraints op.
//

#gpu_target_no_mma = #iree_gpu.target<arch = "gfx942", features = "", wgp = <
  compute = fp32, storage = b32, subgroup = shuffle,
  mma = [],
  subgroup_size_choices = [64],
  max_workgroup_sizes = [1024, 1024, 1024],
  max_thread_count_per_workgroup = 1024,
  max_workgroup_memory_bytes = 65536,
  max_workgroup_counts = [2147483647, 2147483647, 2147483647]
>>
#exec_target_no_mma = #hal.executable.target<"rocm", "rocm-hsaco-fb",
    {iree_codegen.target_info = #gpu_target_no_mma}>

func.func @matmul_no_mma() attributes {hal.executable.target = #exec_target_no_mma} {
  %lhs = tensor.empty() : tensor<128x64xf16>
  %rhs = tensor.empty() : tensor<64x256xf16>
  %empty = tensor.empty() : tensor<128x256xf32>
  %result = linalg.matmul {root_op = #iree_codegen.root_op<set = 0>}
      ins(%lhs, %rhs : tensor<128x64xf16>, tensor<64x256xf16>)
      outs(%empty : tensor<128x256xf32>) -> tensor<128x256xf32>
  return
}

// CHECK-LABEL: func.func @matmul_no_mma
// CHECK:       linalg.matmul
// CHECK-NOT:   iree_codegen.smt.constraints
// CHECK-NOT:   pipeline = #iree_gpu.pipeline<TileAndFuse>

// -----

//
// Test 4: Convolution does NOT trigger the TaF emitter in v0
// (contraction-only gate). VD still emits its own constraints op.
//

#gpu_target_conv = #iree_gpu.target<arch = "gfx942", features = "", wgp = <
  compute = fp32, storage = b32, subgroup = shuffle,
  mma = [<MFMA_F32_16x16x4_F32>],
  subgroup_size_choices = [64],
  max_load_instruction_bits = 128,
  max_workgroup_sizes = [1024, 1024, 1024],
  max_thread_count_per_workgroup = 1024,
  max_workgroup_memory_bytes = 65536,
  max_workgroup_counts = [2147483647, 2147483647, 2147483647]
>>
#exec_target_conv = #hal.executable.target<"rocm", "rocm-hsaco-fb",
    {iree_codegen.target_info = #gpu_target_conv}>

func.func @conv_2d_nhwc_hwcf()
    attributes {hal.executable.target = #exec_target_conv} {
  %input = tensor.empty() : tensor<1x18x18x64xf32>
  %filter = tensor.empty() : tensor<3x3x64x128xf32>
  %empty = tensor.empty() : tensor<1x16x16x128xf32>
  %result = linalg.conv_2d_nhwc_hwcf {
      dilations = dense<1> : tensor<2xi64>,
      root_op = #iree_codegen.root_op<set = 0>,
      strides = dense<1> : tensor<2xi64>}
      ins(%input, %filter : tensor<1x18x18x64xf32>,
                              tensor<3x3x64x128xf32>)
      outs(%empty : tensor<1x16x16x128xf32>) -> tensor<1x16x16x128xf32>
  return
}

// CHECK-LABEL: func.func @conv_2d_nhwc_hwcf
// CHECK:       linalg.conv_2d_nhwc_hwcf
// CHECK-NOT:   pipeline = #iree_gpu.pipeline<TileAndFuse>

// -----

//
// Test 5: Sub-byte element types (i4) cause the TaF emitter to bail out
// (Issue #1 from PR #24484 review — bytes-per-element rounds to 0 and
// would silently void the shared-memory constraint). VD may still emit.
//

#gpu_target_i4 = #iree_gpu.target<arch = "gfx942", features = "", wgp = <
  compute = fp32, storage = b32, subgroup = shuffle,
  mma = [<MFMA_I32_16x16x32_I8>],
  subgroup_size_choices = [64],
  max_load_instruction_bits = 128,
  max_workgroup_sizes = [1024, 1024, 1024],
  max_thread_count_per_workgroup = 1024,
  max_workgroup_memory_bytes = 65536,
  max_workgroup_counts = [2147483647, 2147483647, 2147483647]
>>
#exec_target_i4 = #hal.executable.target<"rocm", "rocm-hsaco-fb",
    {iree_codegen.target_info = #gpu_target_i4}>

func.func @matmul_i4_inputs() attributes {hal.executable.target = #exec_target_i4} {
  %lhs = tensor.empty() : tensor<128x64xi4>
  %rhs = tensor.empty() : tensor<64x256xi4>
  %empty = tensor.empty() : tensor<128x256xi32>
  %result = linalg.matmul {root_op = #iree_codegen.root_op<set = 0>}
      ins(%lhs, %rhs : tensor<128x64xi4>, tensor<64x256xi4>)
      outs(%empty : tensor<128x256xi32>) -> tensor<128x256xi32>
  return
}

// CHECK-LABEL: func.func @matmul_i4_inputs
// CHECK:       linalg.matmul
// CHECK-NOT:   pipeline = #iree_gpu.pipeline<TileAndFuse>
