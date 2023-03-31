// Copyright (c) Microsoft Corporation.
// SPDX-License-Identifier: Apache-2.0

// DeepSpeed Team

#include "custom_cuda_layers.h"
#include "memory_access_utils.h"

namespace cg = cooperative_groups;

namespace td_data {
constexpr int granularity = 16;
}

template <typename T>
__global__ void gather_tokens_impl(T* retained_tokens,
                                   const T* activations,
                                   int32_t* gather_indices,
                                   int32_t sampled_tokens,
                                   int32_t channels,
                                   int32_t read_batch_stride,
                                   int32_t read_seq_stride,
                                   int32_t write_batch_stride,
                                   int32_t write_seq_stride)
{
    constexpr int mem_vals_t = td_data::granularity / sizeof(T);

    cg::thread_block tb = cg::this_thread_block();

    const int gather_idx = gather_indices[tb.group_index().x * sampled_tokens + tb.group_index().y];

    const int read_offset = read_batch_stride * tb.group_index().x + read_seq_stride * gather_idx;
    const int write_offset =
        write_batch_stride * tb.group_index().x + write_seq_stride * tb.group_index().y;

    for (int i = tb.thread_index().x * mem_vals_t; i < channels; i += blockDim.x * mem_vals_t) {
        T local_data[mem_vals_t];
        mem_access::load_global<td_data::granularity>(local_data, activations + read_offset + i);
        mem_access::store_global<td_data::granularity>(retained_tokens + write_offset + i,
                                                       local_data);
    }
}

template <typename T>
void launch_gather_tokens(T* retained_tokens,
                          T* activations,
                          int32_t* gather_indices,
                          int32_t batch_size,
                          int32_t sampled_tokens,
                          int32_t channels,
                          int32_t read_batch_stride,
                          int32_t read_seq_stride,
                          int32_t write_batch_stride,
                          int32_t write_seq_stride,
                          cudaStream_t stream)
{
    constexpr int mem_vals_t = td_data::granularity / sizeof(T);

    const int load_steps = (channels + mem_vals_t - 1) / mem_vals_t;
    const int threads = (load_steps >= 1024) ? 1024 : load_steps;

    dim3 block(threads);
    dim3 grid(batch_size, sampled_tokens);

    gather_tokens_impl<T><<<grid, block, 0, stream>>>(retained_tokens,
                                                      activations,
                                                      gather_indices,
                                                      sampled_tokens,
                                                      channels,
                                                      read_batch_stride,
                                                      read_seq_stride,
                                                      write_batch_stride,
                                                      write_seq_stride);
}

template void launch_gather_tokens<float>(float*,
                                          float*,
                                          int32_t*,
                                          int32_t,
                                          int32_t,
                                          int32_t,
                                          int32_t,
                                          int32_t,
                                          int32_t,
                                          int32_t,
                                          cudaStream_t);

template void launch_gather_tokens<__half>(__half*,
                                           __half*,
                                           int32_t*,
                                           int32_t,
                                           int32_t,
                                           int32_t,
                                           int32_t,
                                           int32_t,
                                           int32_t,
                                           int32_t,
                                           cudaStream_t);

template <typename T>
__global__ void scatter_tokens_impl(T* all_activations,
                                    const T* layer_activations,
                                    int32_t* gather_indices,
                                    int32_t retained_tokens,
                                    int32_t channels,
                                    int32_t read_batch_stride,
                                    int32_t read_seq_stride,
                                    int32_t write_batch_stride,
                                    int32_t write_seq_stride)
{
    constexpr int mem_vals_t = td_data::granularity / sizeof(T);

    cg::thread_block tb = cg::this_thread_block();

    const int gather_idx =
        gather_indices[tb.group_index().x * retained_tokens + tb.group_index().y];

    const int read_offset =
        read_batch_stride * tb.group_index().x + read_seq_stride * tb.group_index().y;
    const int write_offset =
        write_batch_stride * tb.group_index().x + write_seq_stride * gather_idx;

    for (int i = tb.thread_index().x * mem_vals_t; i < channels; i += mem_vals_t * blockDim.x) {
        T local_data[mem_vals_t];
        mem_access::load_global<td_data::granularity>(local_data,
                                                      layer_activations + read_offset + i);
        mem_access::store_global<td_data::granularity>(all_activations + write_offset + i,
                                                       local_data);
    }
}

template <typename T>
void launch_scatter_tokens(T* all_activations,
                           T* layer_activations,
                           int32_t* gather_indices,
                           int32_t batch_size,
                           int32_t sampled_tokens,
                           int32_t channels,
                           int32_t read_batch_stride,
                           int32_t read_seq_stride,
                           int32_t write_batch_stride,
                           int32_t write_seq_stride,
                           cudaStream_t stream)
{
    constexpr int mem_vals_t = td_data::granularity / sizeof(T);

    const int load_steps = (channels + mem_vals_t - 1) / mem_vals_t;
    const int threads = (load_steps >= 1024) ? 1024 : load_steps;

    dim3 block(threads);
    dim3 grid(batch_size, sampled_tokens);

    scatter_tokens_impl<T><<<grid, block, 0, stream>>>(all_activations,
                                                       layer_activations,
                                                       gather_indices,
                                                       sampled_tokens,
                                                       channels,
                                                       read_batch_stride,
                                                       read_seq_stride,
                                                       write_batch_stride,
                                                       write_seq_stride);
}

template void launch_scatter_tokens<float>(float*,
                                           float*,
                                           int32_t*,
                                           int32_t,
                                           int32_t,
                                           int32_t,
                                           int32_t,
                                           int32_t,
                                           int32_t,
                                           int32_t,
                                           cudaStream_t);

template void launch_scatter_tokens<__half>(__half*,
                                            __half*,
                                            int32_t*,
                                            int32_t,
                                            int32_t,
                                            int32_t,
                                            int32_t,
                                            int32_t,
                                            int32_t,
                                            int32_t,
                                            cudaStream_t);
