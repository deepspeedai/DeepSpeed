// Copyright (c) Microsoft Corporation.
// SPDX-License-Identifier: Apache-2.0

// DeepSpeed Team

#include "conversion_utils.h"
#include "ds_kernel_utils.h"
#include "moe_gather.cuh"
#include "reduction_utils.h"
#include "top_k_gating.cuh"
#include "top_k_utils.h"

namespace gather {

constexpr int access_granularity = 16;
constexpr int threads = 256;

}  // namespace gather

template <typename T, int copyUnroll, int N_TOP_K>
__global__ void moe_gather_kernel(T* layer_output,
                                  const T* moe_output,
                                  const float* scores,
                                  const int32_t* mapped_slots,
                                  int32_t* expert_counts,
                                  const int32_t n_channels,
                                  const int32_t n_experts,
                                  const bool normalize_scales)
{
    constexpr int32_t vector_size = gather::access_granularity / sizeof(T);
    constexpr int32_t stride = vector_size * gather::threads;

    const int32_t token_idx = blockIdx.x;
    int32_t token_mapped_slots[N_TOP_K];

    bool all_slots_invalid = true;
    for (int i = 0; i < N_TOP_K; i++) {
        token_mapped_slots[i] = mapped_slots[token_idx * N_TOP_K + i];
        all_slots_invalid &= (token_mapped_slots[i] == gating::unassigned);
    }

    if (token_idx == 0) {
        // Reset expert counts for its next use.
        if (threadIdx.x < n_experts) { expert_counts[threadIdx.x] = 0; }
    }

    if (all_slots_invalid) {
        // This token was not assigned to anything.
        // TODO(cmikeh2): It's possible we want different behavior here moving forward.
        return;
    }

    float token_scores[N_TOP_K];
    for (int i = 0; i < N_TOP_K; i++) { token_scores[i] = scores[token_idx * N_TOP_K + i]; }

    if (normalize_scales) {
        // Normalize the scores so that they sum to 1.
        float sum = 0.0f;
        for (int i = 0; i < N_TOP_K; i++) { sum += token_scores[i]; }

        if (sum > 0.0f) {
            for (int i = 0; i < N_TOP_K; i++) { token_scores[i] /= sum; }
        }
    }

    const int32_t channel_offset = threadIdx.x * vector_size;

    const T* moe_output_bases[N_TOP_K];
#pragma unroll
    for (int i = 0; i < N_TOP_K; i++) {
        moe_output_bases[i] = moe_output + token_mapped_slots[i] * n_channels + channel_offset;
    }

    T* layer_output_base = layer_output + token_idx * n_channels + channel_offset;

#pragma unroll
    for (int i = 0; i < copyUnroll; i++) {
        if (i * stride + channel_offset < n_channels) {
            float accum_buffer[vector_size];
            for (int j = 0; j < vector_size; j++) {
                accum_buffer[j] = reduce::init<reduce::ROpType::Add>();
            }

#pragma unroll
            for (int j = 0; j < N_TOP_K; j++) {
                T reg_buffer[vector_size];
                mem_access::load_global<gather::access_granularity>(
                    reg_buffer, moe_output_bases[j] + i * stride);

#pragma unroll
                for (int k = 0; k < vector_size; k++) {
                    float up_cast = conversion::to<float>(reg_buffer[k]);
                    accum_buffer[k] += up_cast * token_scores[j];
                }
            }

            T store_buffer[vector_size];
#pragma unroll
            for (int j = 0; j < vector_size; j++) {
                store_buffer[j] = conversion::to<T>(accum_buffer[j]);
            }

            mem_access::store_global<gather::access_granularity>(layer_output_base + i * stride,
                                                                 store_buffer);
        }
    }
}

#define LAUNCH_FOR_UNROLL(COUNT)                                                                \
    case COUNT:                                                                                 \
        moe_gather_kernel<T, COUNT, CONST_TOP_K><<<grid, block, 0, stream>>>(layer_output,      \
                                                                             moe_output,        \
                                                                             scores,            \
                                                                             mapped_slots,      \
                                                                             expert_counts,     \
                                                                             n_channels,        \
                                                                             n_experts,         \
                                                                             normalize_scales); \
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
                       const int32_t n_top_k,
                       const bool normalize_scales,
                       cudaStream_t stream)
{
    constexpr int vals_per_unroll = gather::threads * gather::access_granularity / sizeof(T);
    const int copy_unroll = (n_channels + vals_per_unroll - 1) / vals_per_unroll;

    const dim3 block(gather::threads);
    const dim3 grid(n_tokens);

    TOP_K_SWITCH(n_top_k, [&] {
        switch (copy_unroll) {
            LAUNCH_FOR_UNROLL(1)
            LAUNCH_FOR_UNROLL(2)
            LAUNCH_FOR_UNROLL(3)
            LAUNCH_FOR_UNROLL(4)
            LAUNCH_FOR_UNROLL(5)
            LAUNCH_FOR_UNROLL(6)
        }
    });
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
                                          const int32_t n_top_k,       \
                                          const bool normalize_scales, \
                                          cudaStream_t stream);

INSTANTIATE_GATHER_FOR_TYPE(__half)

#ifdef BF16_AVAILABLE
INSTANTIATE_GATHER_FOR_TYPE(__nv_bfloat16)
#endif
