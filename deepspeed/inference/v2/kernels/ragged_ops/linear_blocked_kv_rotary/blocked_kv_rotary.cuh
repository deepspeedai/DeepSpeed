// Copyright (c) Microsoft Corporation.
// SPDX-License-Identifier: Apache-2.0

// DeepSpeed Team

#pragma once

#include "ds_kernel_utils.h"
#include "ragged_dtypes.h"

#ifdef BF16_AVAILABLE
#include <cuda_bf16.h>
#endif

template <typename T>
void launch_kv_rotary_kernel(T* kv_cache,
                             T* q,
                             T* k,
                             T* v,
                             T* inv_freq,
                             const int32_t rotary_dim,
                             const float theta_base,
                             const BatchWrapperCPP batch_desc,
                             const int qkv_stride,
                             const int kv_cache_stride,
                             const int v_offset,
                             const int inv_freq_stride,
                             const int q_ratio,
                             const int head_size,
                             const int n_tokens,
                             const int n_q_heads,
                             cudaStream_t stream);

template <typename T>
void launch_kv_copy_kernel(T* kv_cache,
                           T* q,
                           T* k,
                           T* v,
                           const BatchWrapperCPP batch_desc,
                           const int qkv_stride,
                           const int kv_cache_stride,
                           const int v_offset,
                           const int q_ratio,
                           const int head_size,
                           const int n_tokens,
                           const int n_q_heads,
                           cudaStream_t stream);
