// Copyright (c) Microsoft Corporation.
// SPDX-License-Identifier: Apache-2.0

// DeepSpeed Team

#include "top_k_gating.h"
#include <c10/cuda/CUDAStream.h>

#define DISPATCH_TOP_K_GATING(T_TYPE, C_TYPE)                   \
    if (logits.options().dtype() == torch::T_TYPE) {            \
        launch_top_k_gating((int32_t*)expert_counts.data_ptr(), \
                            (float*)scores.data_ptr(),          \
                            (int32_t*)assignments.data_ptr(),   \
                            (int32_t*)offsets.data_ptr(),       \
                            (const C_TYPE*)logits.data_ptr(),   \
                            batch_metadata_ptr,                 \
                            n_tokens,                           \
                            n_experts,                          \
                            n_top_k,                            \
                            at::cuda::getCurrentCUDAStream());  \
        return;                                                 \
    }

/*
Perform softmax plus atomics in order to do first pass of top_k_gating.
*/
void top_k_gating(torch::Tensor& expert_counts,
                  torch::Tensor& scores,
                  torch::Tensor& assignments,
                  torch::Tensor& offsets,
                  torch::Tensor& logits,
                  torch::Tensor& batch_metadata)
{
    const int32_t n_tokens = scores.size(0);
    const int32_t n_top_k = scores.size(1);

    // Should have the same buffer size for scores, offsets, and assignments
    TORCH_CHECK(n_tokens == offsets.size(0));
    TORCH_CHECK(n_tokens == logits.size(0));
    TORCH_CHECK(n_tokens == assignments.size(0));

    TORCH_CHECK(n_top_k == offsets.size(1));
    TORCH_CHECK(n_top_k == assignments.size(1));

    TORCH_CHECK(expert_counts.scalar_type() == torch::kInt32);
    TORCH_CHECK(scores.scalar_type() == torch::kFloat);
    TORCH_CHECK(assignments.scalar_type() == torch::kInt32);
    TORCH_CHECK(offsets.scalar_type() == torch::kInt32);

    const int32_t n_experts = logits.size(1);
    const RaggedBatchDescriptor* batch_metadata_ptr =
        reinterpret_cast<const RaggedBatchDescriptor*>(batch_metadata.data_ptr());

    DISPATCH_TOP_K_GATING(kFloat, float)
    DISPATCH_TOP_K_GATING(kHalf, __half)
#ifdef BF16_AVAILABLE
    DISPATCH_TOP_K_GATING(kBFloat16, __nv_bfloat16)
#endif

    TORCH_CHECK(false, "Unsupported dtype for logits in top_k_gating");
}
