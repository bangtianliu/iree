// E2E tests for arg_compare with GPU VectorDistribute pipeline
// These tests specifically target the ballot-optimized reduction path

// Test 1: Basic argmax with power-of-2 size (matches subgroup size)
func.func @arg_compare_64_elements_max() {
  // 64 elements - fits exactly in a typical AMD subgroup
  %input_values = util.unfoldable_constant dense<[
    1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0,
    9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
    17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0,
    25.0, 26.0, 27.0, 28.0, 29.0, 30.0, 31.0, 32.0,
    33.0, 34.0, 35.0, 36.0, 37.0, 38.0, 39.0, 40.0,
    41.0, 42.0, 43.0, 44.0, 45.0, 46.0, 47.0, 48.0,
    49.0, 50.0, 51.0, 52.0, 53.0, 54.0, 55.0, 56.0,
    57.0, 58.0, 59.0, 60.0, 61.0, 62.0, 63.0, 64.0
  ]> : tensor<64xf32>

  %out_value_empty = tensor.empty() : tensor<f32>
  %out_index_empty = tensor.empty() : tensor<i32>

  %neg_inf = arith.constant 0xFF800000 : f32
  %c0 = arith.constant 0 : i32
  %out_value = linalg.fill ins(%neg_inf : f32) outs(%out_value_empty : tensor<f32>) -> tensor<f32>
  %out_index = linalg.fill ins(%c0 : i32) outs(%out_index_empty : tensor<i32>) -> tensor<i32>

  %0:2 = iree_linalg_ext.arg_compare
    dimension(0)
    ins(%input_values : tensor<64xf32>)
    outs(%out_value, %out_index : tensor<f32>, tensor<i32>) {
    ^bb0(%a: f32, %b: f32):
      %cmp = arith.cmpf ogt, %a, %b : f32
      iree_linalg_ext.yield %cmp : i1
  } -> tensor<f32>, tensor<i32>

  check.expect_almost_eq_const(
    %0#0,
    dense<64.0> : tensor<f32>
  ) : tensor<f32>

  check.expect_eq_const(
    %0#1,
    dense<63> : tensor<i32>
  ) : tensor<i32>

  return
}

// Test 2: Argmax with tie-breaking (tests ballot finds lowest thread ID)
func.func @arg_compare_ties_prefer_smallest_index() {
  // Multiple occurrences of max value - should select first one
  %input_values = util.unfoldable_constant dense<[
    1.0, 2.0, 10.0, 3.0, 10.0, 4.0, 10.0, 5.0,
    6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0
  ]> : tensor<16xf32>

  %out_value_empty = tensor.empty() : tensor<f32>
  %out_index_empty = tensor.empty() : tensor<i32>

  %neg_inf = arith.constant 0xFF800000 : f32
  %c0 = arith.constant 0 : i32
  %out_value = linalg.fill ins(%neg_inf : f32) outs(%out_value_empty : tensor<f32>) -> tensor<f32>
  %out_index = linalg.fill ins(%c0 : i32) outs(%out_index_empty : tensor<i32>) -> tensor<i32>

  %0:2 = iree_linalg_ext.arg_compare
    dimension(0)
    ins(%input_values : tensor<16xf32>)
    outs(%out_value, %out_index : tensor<f32>, tensor<i32>) {
    ^bb0(%a: f32, %b: f32):
      %cmp = arith.cmpf ogt, %a, %b : f32
      iree_linalg_ext.yield %cmp : i1
  } -> tensor<f32>, tensor<i32>

  check.expect_almost_eq_const(
    %0#0,
    dense<13.0> : tensor<f32>
  ) : tensor<f32>

  // Should return 15 (the last 13.0), demonstrating ballot's cttz behavior
  check.expect_eq_const(
    %0#1,
    dense<15> : tensor<i32>
  ) : tensor<i32>

  return
}

// Test 3: Argmin test
func.func @arg_compare_128_elements_min() {
  // 128 elements - tests cross-subgroup reduction
  %input_values = util.unfoldable_constant dense<[
    128.0, 127.0, 126.0, 125.0, 124.0, 123.0, 122.0, 121.0,
    120.0, 119.0, 118.0, 117.0, 116.0, 115.0, 114.0, 113.0,
    112.0, 111.0, 110.0, 109.0, 108.0, 107.0, 106.0, 105.0,
    104.0, 103.0, 102.0, 101.0, 100.0, 99.0, 98.0, 97.0,
    96.0, 95.0, 94.0, 93.0, 92.0, 91.0, 90.0, 89.0,
    88.0, 87.0, 86.0, 85.0, 84.0, 83.0, 82.0, 81.0,
    80.0, 79.0, 78.0, 77.0, 76.0, 75.0, 74.0, 73.0,
    72.0, 71.0, 70.0, 69.0, 68.0, 67.0, 66.0, 65.0,
    64.0, 63.0, 62.0, 61.0, 60.0, 59.0, 58.0, 57.0,
    56.0, 55.0, 54.0, 53.0, 52.0, 51.0, 50.0, 49.0,
    48.0, 47.0, 46.0, 45.0, 44.0, 43.0, 42.0, 41.0,
    40.0, 39.0, 38.0, 37.0, 36.0, 35.0, 34.0, 33.0,
    32.0, 31.0, 30.0, 29.0, 28.0, 27.0, 26.0, 25.0,
    24.0, 23.0, 22.0, 21.0, 20.0, 19.0, 18.0, 17.0,
    16.0, 15.0, 14.0, 13.0, 12.0, 11.0, 10.0, 9.0,
    8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0
  ]> : tensor<128xf32>

  %out_value_empty = tensor.empty() : tensor<f32>
  %out_index_empty = tensor.empty() : tensor<i32>

  %pos_inf = arith.constant 0x7F800000 : f32
  %c0 = arith.constant 0 : i32
  %out_value = linalg.fill ins(%pos_inf : f32) outs(%out_value_empty : tensor<f32>) -> tensor<f32>
  %out_index = linalg.fill ins(%c0 : i32) outs(%out_index_empty : tensor<i32>) -> tensor<i32>

  %0:2 = iree_linalg_ext.arg_compare
    dimension(0)
    ins(%input_values : tensor<128xf32>)
    outs(%out_value, %out_index : tensor<f32>, tensor<i32>) {
    ^bb0(%a: f32, %b: f32):
      %cmp = arith.cmpf olt, %a, %b : f32
      iree_linalg_ext.yield %cmp : i1
  } -> tensor<f32>, tensor<i32>

  check.expect_almost_eq_const(
    %0#0,
    dense<1.0> : tensor<f32>
  ) : tensor<f32>

  check.expect_eq_const(
    %0#1,
    dense<127> : tensor<i32>
  ) : tensor<i32>

  return
}

// Test 4: 2D reduction on last dimension (tests parallel reductions)
func.func @arg_compare_2d_batch_max() {
  // 4 batches x 32 elements each
  // Each batch should find its own max independently
  %input_values = util.unfoldable_constant dense<[
    [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
     17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0, 25.0, 26.0, 27.0, 28.0, 29.0, 30.0, 31.0, 32.0],
    [32.0, 31.0, 30.0, 29.0, 28.0, 27.0, 26.0, 25.0, 24.0, 23.0, 22.0, 21.0, 20.0, 19.0, 18.0, 17.0,
     16.0, 15.0, 14.0, 13.0, 12.0, 11.0, 10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0],
    [10.0, 20.0, 5.0, 15.0, 25.0, 8.0, 18.0, 28.0, 12.0, 22.0, 7.0, 17.0, 27.0, 11.0, 21.0, 6.0,
     16.0, 26.0, 9.0, 19.0, 29.0, 13.0, 23.0, 4.0, 14.0, 24.0, 3.0, 30.0, 2.0, 31.0, 1.0, 32.0],
    [16.0, 16.0, 16.0, 16.0, 16.0, 16.0, 16.0, 16.0, 32.0, 16.0, 16.0, 16.0, 16.0, 16.0, 16.0, 16.0,
     16.0, 16.0, 16.0, 16.0, 16.0, 16.0, 16.0, 16.0, 16.0, 16.0, 16.0, 16.0, 16.0, 16.0, 16.0, 16.0]
  ]> : tensor<4x32xf32>

  %out_value_empty = tensor.empty() : tensor<4xf32>
  %out_index_empty = tensor.empty() : tensor<4xi32>

  %neg_inf = arith.constant 0xFF800000 : f32
  %c0 = arith.constant 0 : i32
  %out_value = linalg.fill ins(%neg_inf : f32) outs(%out_value_empty : tensor<4xf32>) -> tensor<4xf32>
  %out_index = linalg.fill ins(%c0 : i32) outs(%out_index_empty : tensor<4xi32>) -> tensor<4xi32>

  %0:2 = iree_linalg_ext.arg_compare
    dimension(1)
    ins(%input_values : tensor<4x32xf32>)
    outs(%out_value, %out_index : tensor<4xf32>, tensor<4xi32>) {
    ^bb0(%a: f32, %b: f32):
      %cmp = arith.cmpf ogt, %a, %b : f32
      iree_linalg_ext.yield %cmp : i1
  } -> tensor<4xf32>, tensor<4xi32>

  check.expect_almost_eq_const(
    %0#0,
    dense<[32.0, 32.0, 32.0, 32.0]> : tensor<4xf32>
  ) : tensor<4xf32>

  check.expect_eq_const(
    %0#1,
    dense<[31, 0, 31, 8]> : tensor<4xi32>
  ) : tensor<4xi32>

  return
}

// Test 5: FP16 test (tests different element types)
func.func @arg_compare_fp16_max() {
  %input_values = util.unfoldable_constant dense<[
    1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0,
    9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0
  ]> : tensor<16xf16>

  %out_value_empty = tensor.empty() : tensor<f16>
  %out_index_empty = tensor.empty() : tensor<i32>

  %neg_inf = arith.constant 0xFC00 : f16  // -inf in fp16
  %c0 = arith.constant 0 : i32
  %out_value = linalg.fill ins(%neg_inf : f16) outs(%out_value_empty : tensor<f16>) -> tensor<f16>
  %out_index = linalg.fill ins(%c0 : i32) outs(%out_index_empty : tensor<i32>) -> tensor<i32>

  %0:2 = iree_linalg_ext.arg_compare
    dimension(0)
    ins(%input_values : tensor<16xf16>)
    outs(%out_value, %out_index : tensor<f16>, tensor<i32>) {
    ^bb0(%a: f16, %b: f16):
      %cmp = arith.cmpf ogt, %a, %b : f16
      iree_linalg_ext.yield %cmp : i1
  } -> tensor<f16>, tensor<i32>

  check.expect_almost_eq_const(
    %0#0,
    dense<16.0> : tensor<f16>
  ) : tensor<f16>

  check.expect_eq_const(
    %0#1,
    dense<15> : tensor<i32>
  ) : tensor<i32>

  return
}
