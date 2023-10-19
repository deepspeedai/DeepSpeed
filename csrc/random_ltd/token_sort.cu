// Copyright (c) Microsoft Corporation.
// SPDX-License-Identifier: Apache-2.0

// DeepSpeed Team

#include <cassert>
#include "custom_cuda_layers.h"
#include "memory_access_utils.h"

namespace cg = cooperative_groups;

namespace td_sort {
constexpr int threads = 512;
constexpr int granularity = 16;
constexpr int mem_vals = granularity / sizeof(int32_t);
constexpr int max_buffer_size = (threads + 1) * mem_vals;

#ifdef __HIP_PLATFORM_AMD__
constexpr int warp_size = 64;
#else
constexpr int warp_size = 32;
#endif

constexpr int max_warps = threads / warp_size;
}  // namespace td_sort

template <int VALS_PER_THREAD>
__global__ void scan_sort(int32_t* data, int reserved_tokens, int original_tokens)
{
    cg::thread_block tb = cg::this_thread_block();
    cg::thread_block_tile<td_sort::warp_size> warp = cg::tiled_partition<td_sort::warp_size>(tb);

    __shared__ int32_t indices_buffer[td_sort::max_buffer_size];
    __shared__ int32_t intermediate_buffer[td_sort::max_warps];
    __shared__ int32_t sorted_indices_buffer[td_sort::max_buffer_size];

    for (int i = tb.thread_index().x * td_sort::mem_vals; i < original_tokens + 1;
         i += tb.group_dim().x * td_sort::mem_vals) {
        uint32_t zeros[td_sort::mem_vals] = {0, 0, 0, 0};
        mem_access::store_shared<td_sort::granularity>(indices_buffer + i, zeros);
    }

    int32_t local_vals[VALS_PER_THREAD];

    // We flatten layers/batch into a single indexing dimension
    int32_t* data_block = data + tb.group_index().x * reserved_tokens;

    // The next two loops really could be fused for a more logical code layout, but don't want to
    // move the barrier forward
#pragma unroll
    for (int i = 0; i < VALS_PER_THREAD; i++) {
        const int iter_idx = i * td_sort::threads + tb.thread_index().x;
        if (iter_idx < reserved_tokens) {
            mem_access::load_global<sizeof(int32_t)>(local_vals + i, data_block + iter_idx);
        } else {
            local_vals[i] = 0;
        }
    }

    tb.sync();

#pragma unroll
    for (int i = 0; i < VALS_PER_THREAD; i++) {
        const int iter_idx = i * td_sort::threads + tb.thread_index().x;
        if (iter_idx < reserved_tokens) {
            const int32_t one = 1;
            mem_access::store_shared<sizeof(int32_t)>(indices_buffer + local_vals[i], &one);
        }
    }

    tb.sync();

    int32_t local_input[td_sort::mem_vals];
    mem_access::load_shared<td_sort::granularity>(
        local_input, indices_buffer + tb.thread_index().x * td_sort::mem_vals);

    int32_t reduce_vals[td_sort::mem_vals];
    reduce_vals[0] = local_input[0];

#pragma unroll
    for (int i = 1; i < td_sort::mem_vals; i++) {
        reduce_vals[i] = local_input[i] + reduce_vals[i - 1];
    }

    int32_t step_1_val = reduce_vals[td_sort::mem_vals - 1];
    // Short span exclusive scan algorithm (less work efficient)
#pragma unroll
    for (int i = 1; i < td_sort::warp_size; i *= 2) {
        int32_t step_val = warp.shfl_up(step_1_val, i);
        step_1_val = (warp.thread_rank() < i) ? step_1_val : step_1_val + step_val;
    }

    if (warp.thread_rank() == td_sort::warp_size - 1) {
        mem_access::store_shared<sizeof(int32_t)>(intermediate_buffer + warp.meta_group_rank(),
                                                  &step_1_val);
    }

    tb.sync();

    if (warp.meta_group_rank() == 0) {
        int32_t step_2_val = 0;
        if (warp.thread_rank() < td_sort::max_warps) {
            mem_access::load_shared<sizeof(int32_t)>(&step_2_val,
                                                     intermediate_buffer + warp.thread_rank());
        }

#pragma unroll
        for (int i = 1; i < td_sort::warp_size; i *= 2) {
            int32_t step_val = warp.shfl_up(step_2_val, i);
            step_2_val = (warp.thread_rank() < i) ? step_2_val : step_2_val + step_val;
        }

        if (warp.thread_rank() < td_sort::max_warps) {
            mem_access::store_shared<sizeof(int32_t)>(intermediate_buffer + warp.thread_rank(),
                                                      &step_2_val);
        }
    }

    tb.sync();

    int step_2_val = 0;
    if (warp.meta_group_rank() > 0) {
        mem_access::load_shared<sizeof(int32_t)>(&step_2_val,
                                                 intermediate_buffer + warp.meta_group_rank() - 1);
    }

    const int thread_offset = reduce_vals[td_sort::mem_vals - 1];

#pragma unroll
    for (int i = 0; i < td_sort::mem_vals; i++) {
        reduce_vals[i] += step_1_val + step_2_val - thread_offset;
    }
    mem_access::store_shared<td_sort::granularity>(
        indices_buffer + tb.thread_index().x * td_sort::mem_vals, reduce_vals);

    if (tb.thread_index().x == 0) {
        indices_buffer[original_tokens] = original_tokens - indices_buffer[original_tokens];
    }
    tb.sync();

    for (int i = 0; i < VALS_PER_THREAD; i++) {
        const int iter_idx = i * td_sort::threads + tb.thread_index().x;
        if (iter_idx < reserved_tokens) {
            if (local_vals[i] == 0) {
                int zero = 0;
                mem_access::store_shared<sizeof(int32_t)>(sorted_indices_buffer, &zero);
            } else {
                int sorted_idx;
                mem_access::load_shared<sizeof(int32_t)>(&sorted_idx,
                                                         indices_buffer + local_vals[i] - 1);
                mem_access::store_shared<sizeof(int32_t)>(sorted_indices_buffer + sorted_idx,
                                                          local_vals + i);
            }
        }
    }

    tb.sync();

#pragma unroll
    for (int i = 0; i < VALS_PER_THREAD; i++) {
        const int iter_idx = i * td_sort::threads + tb.thread_index().x;
        if (iter_idx < reserved_tokens) {
            int32_t store_val;
            mem_access::load_shared<sizeof(int32_t)>(&store_val, sorted_indices_buffer + iter_idx);
            mem_access::store_global<sizeof(int32_t)>(data_block + iter_idx, &store_val);
        }
    }
}

void launch_token_sort(int32_t* indices,
                       int layers,
                       int batch_size,
                       int reserved_size,
                       int original_tokens,
                       cudaStream_t stream)
{
    // Each sort is completely independent, can flatten this dimension
    dim3 grid(layers * batch_size);
    dim3 block(td_sort::threads);

    const int vals_per_thread = (reserved_size + td_sort::threads - 1) / td_sort::threads;

    if (vals_per_thread == 1) {
        scan_sort<1><<<grid, block, 0, stream>>>(indices, reserved_size, original_tokens);
    } else if (vals_per_thread == 2) {
        scan_sort<2><<<grid, block, 0, stream>>>(indices, reserved_size, original_tokens);
    } else if (vals_per_thread == 3) {
        scan_sort<3><<<grid, block, 0, stream>>>(indices, reserved_size, original_tokens);
    } else if (vals_per_thread == 4) {
        scan_sort<4><<<grid, block, 0, stream>>>(indices, reserved_size, original_tokens);
    } else {
        assert(false);
    }
}
