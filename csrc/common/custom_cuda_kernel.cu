// Copyright (c) Microsoft Corporation.
// SPDX-License-Identifier: Apache-2.0

// DeepSpeed Team

#include "custom_cuda_layers.h"

__global__ void param_update_kernel(const float* input, __half* output, int size)
{
    int id = blockIdx.x * blockDim.x + threadIdx.x;

    if (id < size) { output[id] = (__half)input[id]; }
}

void launch_param_update(const float* input, __half* output, int size, cudaStream_t stream)
{
    int threads = 1024;

    dim3 grid_dim((size - 1) / threads + 1);
    dim3 block_dim(threads);

    param_update_kernel<<<grid_dim, block_dim, 0, stream>>>(input, output, size);
}

__global__ void param_update_kernel_half(const float* input, __half* output, int size)
{
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    __half2* output_cast = reinterpret_cast<__half2*>(output);
    if (id < size) {
        float input_f = input[id];
        __half2* input_h = reinterpret_cast<__half2*>(&input_f);
        output_cast[id] = *input_h;
    }
}

void launch_param_update_half(const float* input, __half* output, int size, cudaStream_t stream)
{
    int threads = 1024;
    size /= 2;
    dim3 grid_dim((size - 1) / threads + 1);
    dim3 block_dim(threads);

    param_update_kernel_half<<<grid_dim, block_dim, 0, stream>>>(input, output, size);
}
