// RUN: iree-opt --split-input-file \
// RUN:   --iree-codegen-experimental-verify-pipeline-constraints \
// RUN:   --pass-pipeline='builtin.module(func.func(iree-codegen-insert-smt-constraints,iree-codegen-verify-smt-constraints))' %s \
// RUN:   --verify-diagnostics \
// RUN:   | FileCheck %s

// Test: End-to-end failure from generated constraints.
// This ensures constraints are inserted and that verification reports violations.
// It also catches cases where incorrect knob templates skip verification.
#gpu_target = #iree_gpu.target<arch = "gfx942", features = "", wgp = <
  compute = fp32, storage = b32, subgroup = shuffle,
  mma = [<MFMA_F32_16x16x4_F32>],
  subgroup_size_choices = [64],
  max_load_instruction_bits = 128,
  max_workgroup_sizes = [1024, 1024, 1024], max_thread_count_per_workgroup = 1024,
  max_workgroup_memory_bytes = 65536,
  max_workgroup_counts = [2147483647, 2147483647, 2147483647]
>>
#exec_target = #hal.executable.target<"rocm", "rocm-hsaco-fb",
    {iree_codegen.target_info = #gpu_target}>
#translation = #iree_codegen.translation_info<
    pipeline = #iree_gpu.pipeline<VectorDistribute>
    workgroup_size = [64, 1, 1] subgroup_size = 64>

func.func @matmul_e2e_generated_violation(
    %lhs: tensor<128x64xf32>, %rhs: tensor<64x256xf32>)
    -> tensor<128x256xf32>
    attributes {hal.executable.target = #exec_target,
                translation_info = #translation} {
  %cst = arith.constant 0.0 : f32
  %init = tensor.empty() : tensor<128x256xf32>
  %fill = linalg.fill {root_op = #iree_codegen.root_op<set = 0>}
      ins(%cst : f32) outs(%init : tensor<128x256xf32>)
      -> tensor<128x256xf32>
  // expected-error @below {{pipeline constraints violated}}
  // expected-note @below {{dim_0 must be divisible by wg_0 (128 % 48 == 0)}}
  %result = linalg.matmul {
      lowering_config = #iree_gpu.lowering_config<{
          workgroup = [48, 64, 0],
          reduction = [0, 0, 16],
          mma_kind = #iree_gpu.mma_layout<MFMA_F32_16x16x4_F32>,
          subgroup_basis = [[1, 1, 1], [0, 1, 2]]}>,
      root_op = #iree_codegen.root_op<set = 0>}
      ins(%lhs, %rhs : tensor<128x64xf32>, tensor<64x256xf32>)
      outs(%fill : tensor<128x256xf32>) -> tensor<128x256xf32>
  return %result : tensor<128x256xf32>
}

// -----

#gpu_target = #iree_gpu.target<arch = "gfx942", features = "", wgp = <
  compute = fp32, storage = b32, subgroup = shuffle,
  mma = [<MFMA_F32_16x16x4_F32>],
  subgroup_size_choices = [64],
  max_load_instruction_bits = 128,
  max_workgroup_sizes = [1024, 1024, 1024], max_thread_count_per_workgroup = 1024,
  max_workgroup_memory_bytes = 65536,
  max_workgroup_counts = [2147483647, 2147483647, 2147483647]
>>
#exec_target = #hal.executable.target<"rocm", "rocm-hsaco-fb",
    {iree_codegen.target_info = #gpu_target}>
#translation = #iree_codegen.translation_info<
    pipeline = #iree_gpu.pipeline<VectorDistribute>
    workgroup_size = [64, 1, 1] subgroup_size = 64>

func.func @conv_e2e_generated_violation(
    %input: tensor<1x18x130x64xf32>, %filter: tensor<3x3x64x128xf32>)
    -> tensor<1x16x128x128xf32>
    attributes {hal.executable.target = #exec_target,
                translation_info = #translation} {
  %cst = arith.constant 0.0 : f32
  %init = tensor.empty() : tensor<1x16x128x128xf32>
  %fill = linalg.fill {root_op = #iree_codegen.root_op<set = 1>}
      ins(%cst : f32) outs(%init : tensor<1x16x128x128xf32>)
      -> tensor<1x16x128x128xf32>
  // expected-error @below {{pipeline constraints violated}}
  // expected-note @below {{dim_2 must be divisible by wg_2 (128 % 48 == 0)}}
  %result = linalg.conv_2d_nhwc_hwcf {
      dilations = dense<1> : tensor<2xi64>,
      lowering_config = #iree_gpu.lowering_config<{
          workgroup = [1, 1, 48, 64, 0, 0, 0],
          reduction = [0, 0, 0, 0, 1, 1, 16],
          mma_kind = #iree_gpu.mma_layout<MFMA_F32_16x16x4_F32>,
          subgroup_basis = [[1, 1, 1, 1, 1, 1, 1],
                            [0, 1, 2, 3, 4, 5, 6]]}>,
      root_op = #iree_codegen.root_op<set = 1>,
      strides = dense<1> : tensor<2xi64>}
      ins(%input, %filter : tensor<1x18x130x64xf32>,
                               tensor<3x3x64x128xf32>)
      outs(%fill : tensor<1x16x128x128xf32>) -> tensor<1x16x128x128xf32>
  return %result : tensor<1x16x128x128xf32>
}

// -----

// Test: End-to-end constraint insertion and verification.
// Use the same shapes as above but with divisible workgroup sizes.
// It should pass verification and have constraints erased.
#gpu_target = #iree_gpu.target<arch = "gfx942", features = "", wgp = <
  compute = fp32, storage = b32, subgroup = shuffle,
  mma = [<MFMA_F32_16x16x4_F32>],
  subgroup_size_choices = [64],
  max_load_instruction_bits = 128,
  max_workgroup_sizes = [1024, 1024, 1024], max_thread_count_per_workgroup = 1024,
  max_workgroup_memory_bytes = 65536,
  max_workgroup_counts = [2147483647, 2147483647, 2147483647]
>>
#exec_target = #hal.executable.target<"rocm", "rocm-hsaco-fb",
    {iree_codegen.target_info = #gpu_target}>
#translation = #iree_codegen.translation_info<
    pipeline = #iree_gpu.pipeline<VectorDistribute>
    workgroup_size = [64, 1, 1] subgroup_size = 64>

func.func @matmul_e2e_constraints_erased(
    %lhs: tensor<128x64xf32>, %rhs: tensor<64x256xf32>)
    -> tensor<128x256xf32>
    attributes {hal.executable.target = #exec_target,
                translation_info = #translation} {
  %cst = arith.constant 0.0 : f32
  %init = tensor.empty() : tensor<128x256xf32>
  %fill = linalg.fill {root_op = #iree_codegen.root_op<set = 0>}
      ins(%cst : f32) outs(%init : tensor<128x256xf32>)
      -> tensor<128x256xf32>
  %result = linalg.matmul {
      lowering_config = #iree_gpu.lowering_config<{
          workgroup = [32, 64, 0],
          reduction = [0, 0, 16],
          mma_kind = #iree_gpu.mma_layout<MFMA_F32_16x16x4_F32>,
          subgroup_basis = [[1, 1, 1], [0, 1, 2]]}>,
      root_op = #iree_codegen.root_op<set = 0>}
      ins(%lhs, %rhs : tensor<128x64xf32>, tensor<64x256xf32>)
      outs(%fill : tensor<128x256xf32>) -> tensor<128x256xf32>
  return %result : tensor<128x256xf32>
}

// CHECK-LABEL: func.func @matmul_e2e_constraints_erased
// CHECK:       linalg.matmul
// CHECK-NOT:   iree_codegen.smt.constraints

func.func @conv_e2e_constraints_erased(
    %input: tensor<1x18x130x64xf32>, %filter: tensor<3x3x64x128xf32>)
    -> tensor<1x16x128x128xf32>
    attributes {hal.executable.target = #exec_target,
                translation_info = #translation} {
  %cst = arith.constant 0.0 : f32
  %init = tensor.empty() : tensor<1x16x128x128xf32>
  %fill = linalg.fill {root_op = #iree_codegen.root_op<set = 1>}
      ins(%cst : f32) outs(%init : tensor<1x16x128x128xf32>)
      -> tensor<1x16x128x128xf32>
  %result = linalg.conv_2d_nhwc_hwcf {
      dilations = dense<1> : tensor<2xi64>,
      lowering_config = #iree_gpu.lowering_config<{
          workgroup = [1, 1, 64, 64, 0, 0, 0],
          reduction = [0, 0, 0, 0, 1, 1, 16],
          mma_kind = #iree_gpu.mma_layout<MFMA_F32_16x16x4_F32>,
          subgroup_basis = [[1, 1, 1, 1, 1, 1, 1],
                            [0, 1, 2, 3, 4, 5, 6]]}>,
      root_op = #iree_codegen.root_op<set = 1>,
      strides = dense<1> : tensor<2xi64>}
      ins(%input, %filter : tensor<1x18x130x64xf32>,
                               tensor<3x3x64x128xf32>)
      outs(%fill : tensor<1x16x128x128xf32>) -> tensor<1x16x128x128xf32>
  return %result : tensor<1x16x128x128xf32>
}

// CHECK-LABEL: func.func @conv_e2e_constraints_erased
// CHECK:       linalg.conv_2d_nhwc_hwcf
// CHECK-NOT:   iree_codegen.smt.constraints

// -----

// Test: End-to-end TileAndFuse constraint insertion and verification, passing.
// Picks a TaF+MMA matmul whose lowering_config satisfies all v0 constraints:
//   wg=[32, 32, 0], sg=[1, 1, 0], red=[0, 0, 1], MFMA_16x16x16_F16.
//   Derived: sg_m_cnt = sg_n_cnt = 2, sg_num = 4 (matches the tuner's
//   num_subgroups=4 hard pin), wg_size_x = sg_num * sg_size = 4 * 64 = 256.
// Verifier should erase the constraints op without diagnostics.
#gpu_target_taf = #iree_gpu.target<arch = "gfx942", features = "", wgp = <
  compute = fp32, storage = b32, subgroup = shuffle,
  mma = [<MFMA_F32_16x16x16_F16>],
  subgroup_size_choices = [64],
  max_load_instruction_bits = 128,
  max_workgroup_sizes = [1024, 1024, 1024], max_thread_count_per_workgroup = 1024,
  max_workgroup_memory_bytes = 65536,
  max_workgroup_counts = [2147483647, 2147483647, 2147483647]
>>
#exec_target_taf = #hal.executable.target<"rocm", "rocm-hsaco-fb",
    {iree_codegen.target_info = #gpu_target_taf}>
#translation_taf = #iree_codegen.translation_info<
    pipeline = #iree_gpu.pipeline<TileAndFuse>
    workgroup_size = [256, 1, 1] subgroup_size = 64>

func.func @matmul_taf_e2e_constraints_erased(
    %lhs: tensor<128x64xf16>, %rhs: tensor<64x256xf16>)
    -> tensor<128x256xf32>
    attributes {hal.executable.target = #exec_target_taf,
                translation_info = #translation_taf} {
  %cst = arith.constant 0.0 : f32
  %init = tensor.empty() : tensor<128x256xf32>
  %fill = linalg.fill {root_op = #iree_codegen.root_op<set = 0>}
      ins(%cst : f32) outs(%init : tensor<128x256xf32>)
      -> tensor<128x256xf32>
  // reduction = [0, 0, 1] means 1 mma_k slab per iteration (16 elements);
  // K=64 → 4 iterations and (red_k * mma_k) | K.
  %result = linalg.matmul {
      lowering_config = #iree_gpu.lowering_config<{
          workgroup = [32, 32, 0],
          reduction = [0, 0, 1],
          subgroup = [1, 1, 0],
          mma_kind = #iree_gpu.mma_layout<MFMA_F32_16x16x16_F16>,
          promote_operands = [0, 1]}>,
      root_op = #iree_codegen.root_op<set = 0>}
      ins(%lhs, %rhs : tensor<128x64xf16>, tensor<64x256xf16>)
      outs(%fill : tensor<128x256xf32>) -> tensor<128x256xf32>
  return %result : tensor<128x256xf32>
}

// CHECK-LABEL: func.func @matmul_taf_e2e_constraints_erased
// CHECK:       linalg.matmul
// CHECK-NOT:   iree_codegen.smt.constraints

// -----

// Test: End-to-end TileAndFuse constraint insertion and verification,
// violation case. wg=[48, 64, 0] — 48 does not divide 128, and 48 is not
// a multiple of mma_m=16; the verifier surfaces both.
#gpu_target_taf = #iree_gpu.target<arch = "gfx942", features = "", wgp = <
  compute = fp32, storage = b32, subgroup = shuffle,
  mma = [<MFMA_F32_16x16x16_F16>],
  subgroup_size_choices = [64],
  max_load_instruction_bits = 128,
  max_workgroup_sizes = [1024, 1024, 1024], max_thread_count_per_workgroup = 1024,
  max_workgroup_memory_bytes = 65536,
  max_workgroup_counts = [2147483647, 2147483647, 2147483647]
>>
#exec_target_taf = #hal.executable.target<"rocm", "rocm-hsaco-fb",
    {iree_codegen.target_info = #gpu_target_taf}>
#translation_taf = #iree_codegen.translation_info<
    pipeline = #iree_gpu.pipeline<TileAndFuse>
    workgroup_size = [64, 1, 1] subgroup_size = 64>

func.func @matmul_taf_e2e_generated_violation(
    %lhs: tensor<128x64xf16>, %rhs: tensor<64x256xf16>)
    -> tensor<128x256xf32>
    attributes {hal.executable.target = #exec_target_taf,
                translation_info = #translation_taf} {
  %cst = arith.constant 0.0 : f32
  %init = tensor.empty() : tensor<128x256xf32>
  %fill = linalg.fill {root_op = #iree_codegen.root_op<set = 0>}
      ins(%cst : f32) outs(%init : tensor<128x256xf32>)
      -> tensor<128x256xf32>
  // Two structural violations fire: dim divisibility and the
  // (sg_0 * mma_m) decomposition. With wg=[48,64,0] sg=[2,4,0] the
  // derived sg_m_cnt = 48/(2*16) = 1 (truncated) and sg_n_cnt = 1, so
  // sg_num = 1 != 4 → the sg_num pin also fires.
  // expected-error @below {{pipeline constraints violated}}
  // expected-note @below {{dim_0 must be divisible by wg_0 (128 % 48 == 0)}}
  // expected-note @below {{wg_0 must be divisible by (sg_0 * mma_m) (48 % 32 == 0)}}
  // expected-note @below {{sg_num == 4}}
  %result = linalg.matmul {
      lowering_config = #iree_gpu.lowering_config<{
          workgroup = [48, 64, 0],
          reduction = [0, 0, 1],
          subgroup = [2, 4, 0],
          mma_kind = #iree_gpu.mma_layout<MFMA_F32_16x16x16_F16>,
          promote_operands = [0, 1]}>,
      root_op = #iree_codegen.root_op<set = 0>}
      ins(%lhs, %rhs : tensor<128x64xf16>, tensor<64x256xf16>)
      outs(%fill : tensor<128x256xf32>) -> tensor<128x256xf32>
  return %result : tensor<128x256xf32>
}
