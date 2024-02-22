// Copyright (c) Microsoft Corporation.
// SPDX-License-Identifier: Apache-2.0

// DeepSpeed Team

#include "ds_kernel_utils.h"
#include "reduction_utils.h"
#include "top_k_gating.cuh"
#include "top_k_utils.h"

using ROp = reduce::ROpType;

namespace scatter {

constexpr int access_granularity = 16;
constexpr int threads = 256;
constexpr int warps = threads / hw_warp_size;
constexpr int max_experts = 1024;

}  // namespace scatter

template <typename T, int copyUnroll, int N_TOP_K>
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
    __shared__ int32_t expert_offsets[scatter::max_experts];

    // CG helpers
    cg::thread_block tb = cg::this_thread_block();
    cg::thread_block_tile<hw_warp_size> warp = cg::tiled_partition<hw_warp_size>(tb);

    // Fetch the assigned experts for this token.
    int assigned_experts[N_TOP_K];
    for (int i = 0; i < N_TOP_K; i++) {
        assigned_experts[i] = assignments[token_idx * N_TOP_K + i];
    }

    bool all_unassigned = true;
    for (int i = 0; i < N_TOP_K; i++) {
        if (assigned_experts[i] != gating::unassigned) {
            all_unassigned = false;
        } else {
            mapped_slots[token_idx * N_TOP_K + i] = gating::unassigned;
        }
    }
    if (all_unassigned && token_idx != 0) return;

    // Do a prefix scan on the expert counts to get the base offsets. Here we use the
    // single up-sweep variant.
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

    // Token 0 will write the
    if (token_idx == 0 && tidx < n_experts) {
        int64_t expert_cumsum_64 = (int64_t)expert_cumsum;
        expert_count_cumsums[tidx] = expert_cumsum_64;
    }

    // Since token 0 has now written the expert cumsum to global memory,
    // if it has no valid experts, we can early return.
    if (token_idx == 0 && all_unassigned) return;

    if (tidx < n_experts) { expert_offsets[tidx] = expert_cumsum; }

    // Ensure all the expert offsets are written in shared memory.
    tb.sync();

    // Data copy to appropriate location
    const int32_t thread_offset = tidx * vector_size;

    const int32_t base_load_offset = token_idx * n_channels + thread_offset;
    const T* load_base_ptr = activations + base_load_offset;

    int32_t store_rows[N_TOP_K];
    T* store_base_ptrs[N_TOP_K];
#pragma unroll
    for (int i = 0; i < N_TOP_K; i++) {
        const int32_t cur_expert_offset =
            (assigned_experts[i] > 0) ? expert_offsets[assigned_experts[i] - 1] : 0;
        store_rows[i] = cur_expert_offset + offsets[token_idx * N_TOP_K + i];
        const int32_t base_store_offset = store_rows[i] * n_channels + thread_offset;
        store_base_ptrs[i] = moe_input + base_store_offset;
    }

#pragma unroll
    for (int i = 0; i < copyUnroll; i++) {
        T tmp_buf[vector_size];

        if (i * load_stride + thread_offset < n_channels) {
            mem_access::load_global<scatter::access_granularity>(tmp_buf,
                                                                 load_base_ptr + i * load_stride);
#pragma unroll
            for (int j = 0; j < N_TOP_K; j++) {
                mem_access::store_global<scatter::access_granularity>(
                    store_base_ptrs[j] + i * load_stride, tmp_buf);
            }
        }
    }

    if (threadIdx.x == 0) {
        for (int i = 0; i < N_TOP_K; i++) { mapped_slots[token_idx * N_TOP_K + i] = store_rows[i]; }
    }
}

#define LAUNCH_FOR_UNROLL(COUNT)                               \
    case COUNT:                                                \
        moe_scatter_kernel<T, COUNT, CONST_TOP_K>              \
            <<<grid, block, 0, stream>>>(moe_input,            \
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
                        const int32_t n_top_k,
                        cudaStream_t stream)
{
    constexpr int vals_per_unroll = scatter::threads * scatter::access_granularity / sizeof(T);
    const int copy_unroll = (n_channels + vals_per_unroll - 1) / vals_per_unroll;

    const dim3 block(scatter::threads);
    const dim3 grid(n_tokens);

    TOP_K_SWITCH(n_top_k, [&] {
        switch (copy_unroll) {
            LAUNCH_FOR_UNROLL(1);
            LAUNCH_FOR_UNROLL(2);
            LAUNCH_FOR_UNROLL(3);
            LAUNCH_FOR_UNROLL(4);
            LAUNCH_FOR_UNROLL(5);
            LAUNCH_FOR_UNROLL(6);
        }
    });
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
                                           const int32_t,  \
                                           cudaStream_t);

INSTANTIATE_SCATTER_FOR_TYPE(__half);

#ifdef BF16_AVAILABLE
INSTANTIATE_SCATTER_FOR_TYPE(__nv_bfloat16);
#endif
