// Copyright (c) Microsoft Corporation.
// SPDX-License-Identifier: Apache-2.0

// DeepSpeed Team

#pragma once

#include "ds_kernel_utils.h"
#include "ragged_dtypes.h"

template <typename T>
void launch_moe_gather(T* layer_output,
                       const T* moe_output,
                       const float* scores,
                       const int32_t* mapped_slots,
                       int32_t* expert_counts,
                       const int32_t n_channels,
                       const int32_t n_experts,
                       const int32_t n_tokens,
                       const int32_t n_top_k,
                       const bool normalize_scales,
                       cudaStream_t stream);
