// Copyright (c) Microsoft Corporation.
// SPDX-License-Identifier: Apache-2.0

// DeepSpeed Team

#include "moe_scatter.h"
#include <c10/cuda/CUDAStream.h>

#define DISPATCH_MOE_SCATTER(T_TYPE, C_TYPE)                          \
    if (activations.options().dtype() == torch::T_TYPE) {             \
        launch_moe_scatter((C_TYPE*)moe_input.data_ptr(),             \
                           (int64_t*)expert_count_cumsums.data_ptr(), \
                           (int32_t*)mapped_slots.data_ptr(),         \
                           (const C_TYPE*)activations.data_ptr(),     \
                           (const int32_t*)expert_counts.data_ptr(),  \
                           (const int32_t*)assignments.data_ptr(),    \
                           (const int32_t*)offsets.data_ptr(),        \
                           n_channels,                                \
                           n_tokens,                                  \
                           n_experts,                                 \
                           n_top_k,                                   \
                           at::cuda::getCurrentCUDAStream());         \
        return;                                                       \
    }

/*
Performs a cumsum on the expert counts and copies the hidden states to the
appropriate spot to ensure that each experts inputs are contiguous.
*/
void moe_scatter(torch::Tensor& moe_input,
                 torch::Tensor& expert_count_cumsums,
                 torch::Tensor& mapped_slots,
                 torch::Tensor& activations,
                 torch::Tensor& expert_counts,
                 torch::Tensor& assignments,
                 torch::Tensor& offsets)
{
    const int32_t n_tokens = activations.size(0);
    const int32_t n_channels = activations.size(1);
    const int32_t n_top_k = assignments.size(1);

    // Should have a lot of matching buffer sizes here.
    TORCH_CHECK(n_tokens == assignments.size(0));
    TORCH_CHECK(n_tokens == offsets.size(0));
    TORCH_CHECK(n_channels == moe_input.size(1));

    TORCH_CHECK(n_top_k == offsets.size(1));
    TORCH_CHECK(n_top_k * n_tokens == moe_input.size(0));
    TORCH_CHECK(n_top_k == mapped_slots.size(1));

    const int32_t n_experts = expert_count_cumsums.size(0);

    TORCH_CHECK(moe_input.scalar_type() == activations.scalar_type());
    TORCH_CHECK(expert_count_cumsums.scalar_type() == torch::kInt64);
    TORCH_CHECK(mapped_slots.scalar_type() == torch::kInt32);
    TORCH_CHECK(expert_counts.scalar_type() == torch::kInt32);
    TORCH_CHECK(assignments.scalar_type() == torch::kInt32);
    TORCH_CHECK(offsets.scalar_type() == torch::kInt32);

    DISPATCH_MOE_SCATTER(kHalf, __half);

#ifdef BF16_AVAILABLE
    DISPATCH_MOE_SCATTER(kBFloat16, __nv_bfloat16);
#endif

    TORCH_CHECK(false, "Unsupported dtype for moe_scatter")
}
