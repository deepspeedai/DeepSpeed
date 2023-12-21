// Copyright (c) Microsoft Corporation.
// SPDX-License-Identifier: Apache-2.0

// DeepSpeed Team

#pragma once

#include "ds_kernel_utils.h"
#include "ragged_dtypes.h"

namespace gating {
constexpr int unassigned = -1;
}  // namespace gating

template <typename T>
void launch_top_k_gating(int32_t* expert_counts,
                         float* scores,
                         int32_t* assignments,
                         int32_t* offsets,
                         const T* logits,
                         const RaggedBatchDescriptor* batch_metadata,
                         const int32_t n_tokens,
                         const int32_t n_experts,
                         const int32_t n_top_k,
                         cudaStream_t stream);
