// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_MODULES_HAL_INLINE_MODULE_H_
#define IREE_MODULES_HAL_INLINE_MODULE_H_

#include <stdint.h>

#include "iree/base/api.h"
#include "iree/hal/api.h"
#include "iree/modules/hal/debugging.h"
#include "iree/modules/hal/types.h"
#include "iree/vm/api.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

enum iree_hal_inline_module_flag_bits_t {
  IREE_HAL_INLINE_MODULE_FLAG_NONE = 0u,
};
typedef uint32_t iree_hal_inline_module_flags_t;

// Creates the inline HAL module for local execution.
// This provides ABI compatibility with the full HAL implementation in a much
// smaller footprint. The given |device_allocator| will be used for buffer
// allocations.
IREE_API_EXPORT iree_status_t iree_hal_inline_module_create(
    iree_vm_instance_t* instance, iree_hal_inline_module_flags_t flags,
    iree_hal_module_debug_sink_t debug_sink,
    iree_hal_allocator_t* device_allocator, iree_allocator_t host_allocator,
    iree_vm_module_t** out_module);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_MODULES_HAL_INLINE_MODULE_H_
