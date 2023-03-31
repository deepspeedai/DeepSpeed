// Copyright (c) Microsoft Corporation.
// SPDX-License-Identifier: Apache-2.0

// DeepSpeed Team

#include <cassert>
#include "memory_access_utils.h"
#include "spatial_cuda_layers.h"

/*
Fused bias add variants
*/

namespace badd_opt {
constexpr int threads = 256;
constexpr int steps = 2;
constexpr int granularity = 16;
constexpr int vals_per_h = granularity / sizeof(__half);
constexpr int vals_per_h2 = granularity / sizeof(__half2);
constexpr int vals_per_block = threads * steps * vals_per_h;
constexpr int stride = vals_per_h * threads;
}  // namespace badd_opt

__global__ void opt_bias_add(__half* result,
                             const __half* activation,
                             const __half* bias,
                             int seq_len,
                             int channels)
{
    const int id = blockIdx.x * badd_opt::vals_per_block + threadIdx.x * badd_opt::vals_per_h;
    const int stride = badd_opt::vals_per_h * badd_opt::threads;

    for (int i = 0; i < badd_opt::steps; i++) {
        if (id + i * badd_opt::stride < seq_len * channels) {
            __half2 act_buffer[badd_opt::vals_per_h2];
            __half2 bias_buffer[badd_opt::vals_per_h2];

            mem_access::load_global<badd_opt::granularity>(act_buffer,
                                                           activation + id + i * stride);
            mem_access::load_global<badd_opt::granularity>(bias_buffer,
                                                           bias + ((id + i * stride) % channels));

            for (int j = 0; j < badd_opt::vals_per_h2; j++) { act_buffer[j] += bias_buffer[j]; }

            mem_access::store_global<badd_opt::granularity>(result + id + i * stride, act_buffer);
        }
    }
}

__global__ void opt_bias_add_add(__half* result,
                                 const __half* activation,
                                 const __half* bias,
                                 const __half* other,
                                 int seq_len,
                                 int channels)
{
    const int id = blockIdx.x * badd_opt::vals_per_block + threadIdx.x * badd_opt::vals_per_h;
    const int stride = badd_opt::vals_per_h * badd_opt::threads;

    for (int i = 0; i < badd_opt::steps; i++) {
        if (id + i * badd_opt::stride < seq_len * channels) {
            __half2 act_buffer[badd_opt::vals_per_h2];
            __half2 bias_buffer[badd_opt::vals_per_h2];
            __half2 other_buffer[badd_opt::vals_per_h2];

            mem_access::load_global<badd_opt::granularity>(act_buffer,
                                                           activation + id + i * stride);
            mem_access::load_global<badd_opt::granularity>(bias_buffer,
                                                           bias + ((id + i * stride) % channels));
            mem_access::load_global<badd_opt::granularity>(other_buffer, other + id + i * stride);

            for (int j = 0; j < badd_opt::vals_per_h2; j++) {
                act_buffer[j] += bias_buffer[j] + other_buffer[j];
            }

            mem_access::store_global<badd_opt::granularity>(result + id + i * stride, act_buffer);
        }
    }
}

__global__ void opt_bias_add_bias_add(__half* result,
                                      const __half* activation,
                                      const __half* bias,
                                      const __half* other,
                                      const __half* other_bias,
                                      int seq_len,
                                      int channels)
{
    const int id = blockIdx.x * badd_opt::vals_per_block + threadIdx.x * badd_opt::vals_per_h;
    const int stride = badd_opt::vals_per_h * badd_opt::threads;

    for (int i = 0; i < badd_opt::steps; i++) {
        if (id + i * badd_opt::stride < seq_len * channels) {
            __half2 act_buffer[badd_opt::vals_per_h2];
            __half2 bias_buffer[badd_opt::vals_per_h2];
            __half2 other_buffer[badd_opt::vals_per_h2];
            __half2 other_bias_buffer[badd_opt::vals_per_h2];

            mem_access::load_global<badd_opt::granularity>(act_buffer,
                                                           activation + id + i * stride);
            mem_access::load_global<badd_opt::granularity>(bias_buffer,
                                                           bias + ((id + i * stride) % channels));
            mem_access::load_global<badd_opt::granularity>(other_buffer, other + id + i * stride);
            mem_access::load_global<badd_opt::granularity>(
                other_bias_buffer, other_bias + ((id + i * stride) % channels));

            for (int j = 0; j < badd_opt::vals_per_h2; j++) {
                act_buffer[j] =
                    (act_buffer[j] + bias_buffer[j]) + (other_buffer[j] + other_bias_buffer[j]);
            }

            mem_access::store_global<badd_opt::granularity>(result + id + i * stride, act_buffer);
        }
    }
}

void launch_opt_bias_add(__half* result,
                         const __half* activation,
                         const __half* bias,
                         const __half* other,
                         const __half* other_bias,
                         int batch_size,
                         int seq_len,
                         int channels,
                         cudaStream_t stream)
{
    // Should evaluate `true` for reasonable hidden sizes
    assert(channels % badd_opt::vals_per_h == 0);

    const int effective_seq_len = batch_size * seq_len;
    const int vals = effective_seq_len * channels;

    dim3 block(badd_opt::threads);
    dim3 grid((vals + badd_opt::vals_per_block - 1) / badd_opt::vals_per_block);

    if (!other) {
        // We shouldn't have a bias if there's no activation
        assert(!other_bias);

        opt_bias_add<<<grid, block, 0, stream>>>(
            result, activation, bias, effective_seq_len, channels);
    } else if (!other_bias) {
        opt_bias_add_add<<<grid, block, 0, stream>>>(
            result, activation, bias, other, effective_seq_len, channels);
    } else {
        opt_bias_add_bias_add<<<grid, block, 0, stream>>>(
            result, activation, bias, other, other_bias, effective_seq_len, channels);
    }
}
