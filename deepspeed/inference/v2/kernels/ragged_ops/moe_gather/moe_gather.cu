// Copyright (c) Microsoft Corporation.
// SPDX-License-Identifier: Apache-2.0

// DeepSpeed Team

#include "conversion_utils.h"
#include "ds_kernel_utils.h"
#include "moe_gather.cuh"
#include "reduction_utils.h"
#include "top_1_gating.cuh"

namespace gather {

constexpr int access_granularity = 16;
constexpr int threads = 256;

}  // namespace gather

template <typename T, int copyUnroll>
__global__ void moe_gather_kernel(T* layer_output,
                                  const T* moe_output,
                                  const float* scores,
                                  const int32_t* mapped_slots,
                                  int32_t* expert_counts,
                                  const int32_t n_channels,
                                  const int32_t n_experts)
{
    constexpr int32_t vector_size = gather::access_granularity / sizeof(T);
    constexpr int32_t stride = vector_size * gather::threads;

    const int32_t token_idx = blockIdx.x;
    const int32_t mapped_slot = mapped_slots[token_idx];

    if (token_idx == 0) {
        // Reset expert counts for its next use.
        if (threadIdx.x < n_experts) { expert_counts[threadIdx.x] = 0; }
    }

    if (mapped_slot == gating::unassigned) {
        // This token was not assigned.
        // TODO(cmikeh2): It's possible we want different behavior here moving forward.
        return;
    }

    const float score = scores[token_idx];
    const int32_t channel_offset = threadIdx.x * vector_size;

    const T* moe_output_base = moe_output + mapped_slot * n_channels + channel_offset;
    T* layer_output_base = layer_output + token_idx * n_channels + channel_offset;

#pragma unroll
    for (int i = 0; i < copyUnroll; i++) {
        T reg_buffer[vector_size];

        if (i * stride + channel_offset < n_channels) {
            mem_access::load_global<gather::access_granularity>(reg_buffer,
                                                                moe_output_base + i * stride);

#pragma unroll
            for (int j = 0; j < vector_size; j++) {
                // There are accuracy implications of downcasting the score to a 16-bit
                // data type, so we up-convert the input to 32-bit, multiply, and then
                // down-convert back to 16-bit.
                float up_cast = conversion::to<float>(reg_buffer[j]);
                reg_buffer[j] = conversion::to<T>(up_cast * score);
            }

            mem_access::store_global<gather::access_granularity>(layer_output_base + i * stride,
                                                                 reg_buffer);
        }
    }
}

#define LAUNCH_FOR_UNROLL(COUNT)                                                                   \
    case COUNT:                                                                                    \
        moe_gather_kernel<T, COUNT><<<grid, block, 0, stream>>>(                                   \
            layer_output, moe_output, scores, mapped_slots, expert_counts, n_channels, n_experts); \
        break;

template <typename T>
void launch_moe_gather(T* layer_output,
                       const T* moe_output,
                       const float* scores,
                       const int32_t* mapped_slots,
                       int32_t* expert_counts,
                       const int32_t n_channels,
                       const int32_t n_experts,
                       const int32_t n_tokens,
                       cudaStream_t stream)
{
    constexpr int vals_per_unroll = gather::threads * gather::access_granularity / sizeof(T);
    const int copy_unroll = (n_channels + vals_per_unroll - 1) / vals_per_unroll;

    const dim3 block(gather::threads);
    const dim3 grid(n_tokens);

    switch (copy_unroll) {
        LAUNCH_FOR_UNROLL(1)
        LAUNCH_FOR_UNROLL(2)
        LAUNCH_FOR_UNROLL(3)
        LAUNCH_FOR_UNROLL(4)
        LAUNCH_FOR_UNROLL(5)
        LAUNCH_FOR_UNROLL(6)
    }
}

#define INSTANTIATE_GATHER_FOR_TYPE(TYPE)                              \
    template void launch_moe_gather<TYPE>(TYPE * layer_output,         \
                                          const TYPE* moe_output,      \
                                          const float* scores,         \
                                          const int32_t* mapped_slots, \
                                          int32_t* expert_counts,      \
                                          const int32_t n_channels,    \
                                          const int32_t n_experts,     \
                                          const int32_t n_tokens,      \
                                          cudaStream_t stream);

INSTANTIATE_GATHER_FOR_TYPE(__half)

#ifdef BF16_AVAILABLE
INSTANTIATE_GATHER_FOR_TYPE(__nv_bfloat16)
#endif
