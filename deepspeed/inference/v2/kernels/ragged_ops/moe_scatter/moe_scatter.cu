// Copyright (c) Microsoft Corporation.
// SPDX-License-Identifier: Apache-2.0

// DeepSpeed Team

#include "ds_kernel_utils.h"
#include "moe_scatter.cuh"
#include "reduction_utils.h"
#include "top_1_gating.cuh"

using ROp = reduce::ROpType;

namespace scatter {

constexpr int access_granularity = 16;
constexpr int threads = 256;
constexpr int warps = threads / hw_warp_size;

}  // namespace scatter

template <typename T, int copyUnroll>
__global__ void moe_scatter_kernel(T* moe_input,
                                   int64_t* expert_count_cumsums,
                                   int32_t* mapped_slots,
                                   const T* activations,
                                   const int32_t* assignments,
                                   const int32_t* expert_counts,
                                   const int32_t* offsets,
                                   const int32_t n_channels,
                                   const int32_t n_experts)
{
    constexpr int32_t vector_size = scatter::access_granularity / sizeof(T);
    constexpr int32_t load_stride = vector_size * scatter::threads;

    const int32_t token_idx = blockIdx.x;
    const int32_t tidx = threadIdx.x;
    const int32_t warp_rank = tidx / hw_warp_size;

    // Bank aligned and sufficient
    __shared__ int32_t red_buffer[32];
    __shared__ int32_t token_0_row;

    // CG helpers
    cg::thread_block tb = cg::this_thread_block();
    cg::thread_block_tile<hw_warp_size> warp = cg::tiled_partition<hw_warp_size>(tb);

    const int assigned_expert = assignments[token_idx];

    // For the different codepaths, we'll converge on this variable for doing
    // the token copy.
    int32_t token_base_row;

    if (token_idx == 0) {
        // Token 0 will perform a cumsum on the data
        int32_t expert_vals;
        if (tidx < n_experts) {
            expert_vals = expert_counts[tidx];
        } else {
            expert_vals = 0;
        }

#pragma unroll
        for (int i = 1; i < hw_warp_size; i *= 2) {
            int32_t maybe_add = warp.shfl_up(expert_vals, i);
            expert_vals = (warp.thread_rank() < i) ? expert_vals : expert_vals + maybe_add;
        }

        if (warp.thread_rank() == hw_warp_size - 1) {
            mem_access::store_shared<4>(red_buffer + warp_rank, &expert_vals);
        }

        tb.sync();

        int32_t phase_2_val = 0;
        if (warp.thread_rank() < scatter::warps) {
            mem_access::load_shared<4>(&phase_2_val, red_buffer + warp.thread_rank());
        }

#pragma unroll
        for (int i = 1; i < hw_warp_size; i *= 2) {
            int32_t maybe_add = warp.shfl_up(phase_2_val, i);
            phase_2_val = (warp.thread_rank() < i) ? phase_2_val : phase_2_val + maybe_add;
        }

        int warp_offset = 0;
        if (warp_rank > 0) { warp_offset = warp.shfl(phase_2_val, warp_rank - 1); }
        const int32_t expert_cumsum = warp_offset + expert_vals;

        if (tidx < n_experts) {
            int64_t expert_cumsum_64 = (int64_t)expert_cumsum;
            expert_count_cumsums[tidx] = expert_cumsum_64;
        }

        if (assigned_expert == gating::unassigned) return;
        if (assigned_expert - 1 == tidx) token_0_row = expert_cumsum;

        tb.sync();

        if (assigned_expert != 0) {
            token_base_row = token_0_row;
        } else {
            token_base_row = 0;
        }

    } else if (assigned_expert == gating::unassigned) {
        // For whatever reason, don't need to perform the copy, so we'll early return
        // and signal this wasn't mapped with a negative 1.
        if (tidx == 0) mapped_slots[token_idx] = gating::unassigned;
        return;
    } else {
        // For all other valid tokens, we can just do a block-scoped sum.
        if (tidx < assigned_expert) {
            token_base_row = expert_counts[tidx];
        } else {
            token_base_row = 0;
        }

        warp.sync();

        // TODO(cmikeh2): Shouldn't use the internal api.
        reduce::_block<int32_t, scatter::warps, ROp::Add>(tb, warp, &token_base_row);
    }

    // Data copy to appropriate location
    const int32_t thread_offset = tidx * vector_size;

    const int32_t base_load_offset = token_idx * n_channels + thread_offset;
    const T* load_base_ptr = activations + base_load_offset;

    const int32_t store_row = token_base_row + offsets[token_idx];
    const int32_t base_store_offset = store_row * n_channels + thread_offset;
    T* store_base_ptr = moe_input + base_store_offset;

#pragma unroll
    for (int i = 0; i < copyUnroll; i++) {
        T tmp_buf[vector_size];

        if (i * load_stride + thread_offset < n_channels) {
            mem_access::load_global<scatter::access_granularity>(tmp_buf,
                                                                 load_base_ptr + i * load_stride);
            mem_access::store_global<scatter::access_granularity>(store_base_ptr + i * load_stride,
                                                                  tmp_buf);
        }
    }

    if (threadIdx.x == 0) { mapped_slots[token_idx] = store_row; }
}

#define LAUNCH_FOR_UNROLL(COUNT)                                                       \
    case COUNT:                                                                        \
        moe_scatter_kernel<T, COUNT><<<grid, block, 0, stream>>>(moe_input,            \
                                                                 expert_count_cumsums, \
                                                                 mapped_slots,         \
                                                                 activations,          \
                                                                 assignments,          \
                                                                 expert_counts,        \
                                                                 offsets,              \
                                                                 n_channels,           \
                                                                 n_experts);           \
        break;

template <typename T>
void launch_moe_scatter(T* moe_input,
                        int64_t* expert_count_cumsums,
                        int32_t* mapped_slots,
                        const T* activations,
                        const int32_t* expert_counts,
                        const int32_t* assignments,
                        const int32_t* offsets,
                        const int32_t n_channels,
                        const int32_t n_tokens,
                        const int32_t n_experts,
                        cudaStream_t stream)
{
    constexpr int vals_per_unroll = scatter::threads * scatter::access_granularity / sizeof(T);
    const int copy_unroll = (n_channels + vals_per_unroll - 1) / vals_per_unroll;

    const dim3 block(scatter::threads);
    const dim3 grid(n_tokens);

    switch (copy_unroll) {
        LAUNCH_FOR_UNROLL(1);
        LAUNCH_FOR_UNROLL(2);
        LAUNCH_FOR_UNROLL(3);
        LAUNCH_FOR_UNROLL(4);
        LAUNCH_FOR_UNROLL(5);
        LAUNCH_FOR_UNROLL(6);
    }
}

#define INSTANTIATE_SCATTER_FOR_TYPE(TYPE)                 \
    template void launch_moe_scatter<TYPE>(TYPE*,          \
                                           int64_t*,       \
                                           int32_t*,       \
                                           const TYPE*,    \
                                           const int32_t*, \
                                           const int32_t*, \
                                           const int32_t*, \
                                           const int32_t,  \
                                           const int32_t,  \
                                           const int32_t,  \
                                           cudaStream_t);

INSTANTIATE_SCATTER_FOR_TYPE(__half);

#ifdef BF16_AVAILABLE
INSTANTIATE_SCATTER_FOR_TYPE(__nv_bfloat16);
#endif
