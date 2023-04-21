// Copyright (c) Microsoft Corporation.
// SPDX-License-Identifier: Apache-2.0

// DeepSpeed Team

#include "general_kernels.h"

namespace cg = cooperative_groups;

template <typename T>
__global__ void column_sum_reduce(const T* __restrict__ inp,
                                  T* __restrict__ out,
                                  int rows,
                                  int width)
{
    __shared__ float tile[TILE_DIM][TILE_DIM + 1];

    cg::thread_block b = cg::this_thread_block();
    cg::thread_block_tile<TILE_DIM> g = cg::tiled_partition<TILE_DIM>(b);

    int idx = blockDim.x * blockIdx.x + threadIdx.x;

    int y_stride = width * TILE_DIM;

    float localSum = 0;

    // Loop across matrix height
    if (idx < width) {
        int offset = threadIdx.y * width + idx;
        for (int r = threadIdx.y; r < rows; r += TILE_DIM) {
            localSum += (float)inp[offset];
            offset += y_stride;
        }
    }

    tile[threadIdx.x][threadIdx.y] = localSum;

    __syncthreads();

    // Sum the shared buffer.
    float sum = tile[threadIdx.y][threadIdx.x];

#ifndef __STOCHASTIC_MODE__
    __syncthreads();
#endif

    for (int i = 1; i < TILE_DIM; i <<= 1) sum += g.shfl_down(sum, i);

    if (threadIdx.x == 0) {
        int pos = blockIdx.x * TILE_DIM + threadIdx.y;
        if (pos < width) out[pos] = sum;
    }
}

template <typename T>
void launch_fuse_transpose_bias_kernel(const T* inp,
                                       T* out,
                                       int rows,
                                       int cols,
                                       cudaStream_t stream);

template <>
void launch_fuse_transpose_bias_kernel<float>(const float* inp,
                                              float* out,
                                              int rows,
                                              int cols,
                                              cudaStream_t stream)
{
    // assert(rows % TILE_DIM == 0);
    // assert(cols % TILE_DIM == 0);

    dim3 grid_dim((cols - 1) / TILE_DIM + 1);
    dim3 block_dim(TILE_DIM, TILE_DIM);

    column_sum_reduce<float><<<grid_dim, block_dim, 0, stream>>>(inp, out, rows, cols);
}

template <>
void launch_fuse_transpose_bias_kernel<__half>(const __half* inp,
                                               __half* out,
                                               int rows,
                                               int cols,
                                               cudaStream_t stream)
{
    // assert(rows % TILE_DIM == 0);
    // assert(cols % TILE_DIM == 0);

    dim3 grid_dim((cols - 1) / TILE_DIM + 1);
    dim3 block_dim(TILE_DIM, TILE_DIM);

    column_sum_reduce<__half><<<grid_dim, block_dim, 0, stream>>>(inp, out, rows, cols);
}

__global__ void fused_add2_kernel(const int N, float* out, const float* inp1, const float* inp2)
{
    const float4* inp1_4 = reinterpret_cast<const float4*>(inp1);
    const float4* inp2_4 = reinterpret_cast<const float4*>(inp2);
    float4* out_4 = reinterpret_cast<float4*>(out);

    CUDA_1D_KERNEL_LOOP(j, N)
    {
        float4 val;
        float4 inp1_reg = inp1_4[j];
        float4 inp2_reg = inp2_4[j];

        val.x = inp1_reg.x + inp2_reg.x;
        val.y = inp1_reg.y + inp2_reg.y;
        val.z = inp1_reg.z + inp2_reg.z;
        val.w = inp1_reg.w + inp2_reg.w;

        out_4[j] = val;
    }
}

__global__ void fused_add2_kernel(const int N, __half* out, const __half* inp1, const __half* inp2)
{
    float2 inp1_4;
    float2 inp2_4;

    __half2* inp1_h = reinterpret_cast<__half2*>(&inp1_4);
    __half2* inp2_h = reinterpret_cast<__half2*>(&inp2_4);

    const float2* inp1_arr = reinterpret_cast<const float2*>(inp1);
    const float2* inp2_arr = reinterpret_cast<const float2*>(inp2);

    CUDA_1D_KERNEL_LOOP(j, N)
    {
        inp1_4 = inp1_arr[j];
        inp2_4 = inp2_arr[j];

        float2 inp1_h_f_0 = __half22float2(inp1_h[0]);
        float2 inp1_h_f_1 = __half22float2(inp1_h[1]);

        float2 inp2_h_f_0 = __half22float2(inp2_h[0]);
        float2 inp2_h_f_1 = __half22float2(inp2_h[1]);

        inp1_h_f_0.x += inp2_h_f_0.x;
        inp1_h_f_0.y += inp2_h_f_0.y;
        inp1_h_f_1.x += inp2_h_f_1.x;
        inp1_h_f_1.y += inp2_h_f_1.y;

        float2 val_f;
        __half2* val_h = reinterpret_cast<__half2*>(&val_f);

        val_h[0] = __float22half2_rn(inp1_h_f_0);
        val_h[1] = __float22half2_rn(inp1_h_f_1);

        float2* out_4 = reinterpret_cast<float2*>(out);
        out_4[j] = val_f;
    }
}

template <>
void launch_fused_add2<float>(float* out,
                              const float* inp1,
                              const float* inp2,
                              int batch_size,
                              int seq_length,
                              int hidden_dim,
                              cudaStream_t& stream)
{
    int total_count = batch_size * seq_length * hidden_dim / 4;
    dim3 grid_dim = DS_GET_BLOCKS(total_count);  //(batch_size * seq_length);

    dim3 block_dim = DS_CUDA_NUM_THREADS;  //(hidden_dim / 4);

    fused_add2_kernel<<<grid_dim, block_dim, 0, stream>>>(total_count, out, inp1, inp2);
}

template <>
void launch_fused_add2<__half>(__half* out,
                               const __half* inp1,
                               const __half* inp2,
                               int batch_size,
                               int seq_length,
                               int hidden_dim,
                               cudaStream_t& stream)
{
    int total_count = batch_size * seq_length * hidden_dim / 4;
    dim3 grid_dim = DS_GET_BLOCKS(total_count);  //(batch_size * seq_length);

    dim3 block_dim = DS_CUDA_NUM_THREADS;  //(hidden_dim / 4);

    fused_add2_kernel<<<grid_dim, block_dim, 0, stream>>>(total_count, out, inp1, inp2);
}

__global__ void fused_add3_kernel(float* out,
                                  const float* inp1,
                                  const float* inp2,
                                  const float* inp3,
                                  int size,
                                  int row_stride)
{
    int row = blockIdx.x;
    int id = threadIdx.x;

    const float4* inp1_4 = reinterpret_cast<const float4*>(inp1);
    const float4* inp2_4 = reinterpret_cast<const float4*>(inp2);
    const float4* inp3_4 = reinterpret_cast<const float4*>(inp3);

    float4* out_4 = reinterpret_cast<float4*>(out);

    float4 val;
    float4 inp1_reg = inp1_4[row * row_stride + id];
    float4 inp2_reg = inp2_4[row * row_stride + id];
    float4 inp3_reg = inp3_4[row * row_stride + id];

    val.x = inp1_reg.x + inp2_reg.x + inp3_reg.x;
    val.y = inp1_reg.y + inp2_reg.y + inp3_reg.y;
    val.z = inp1_reg.z + inp2_reg.z + inp3_reg.z;
    val.w = inp1_reg.w + inp2_reg.w + inp3_reg.w;

    out_4[row * row_stride + id] = val;
}

__global__ void fused_add3_kernel(__half* out,
                                  const __half* inp1,
                                  const __half* inp2,
                                  const __half* inp3,
                                  int size,
                                  int row_stride)
{
    int row = blockIdx.x;
    int id = threadIdx.x;
    const float2* inp1_arr = reinterpret_cast<const float2*>(inp1);
    const float2* inp2_arr = reinterpret_cast<const float2*>(inp2);
    const float2* inp3_arr = reinterpret_cast<const float2*>(inp3);

    float2 inp1_4 = inp1_arr[row * row_stride + id];
    float2 inp2_4 = inp2_arr[row * row_stride + id];
    float2 inp3_4 = inp3_arr[row * row_stride + id];

    __half2* inp1_h = reinterpret_cast<__half2*>(&inp1_4);
    __half2* inp2_h = reinterpret_cast<__half2*>(&inp2_4);
    __half2* inp3_h = reinterpret_cast<__half2*>(&inp3_4);

    float2 inp1_h_f_0 = __half22float2(inp1_h[0]);
    float2 inp1_h_f_1 = __half22float2(inp1_h[1]);

    float2 inp2_h_f_0 = __half22float2(inp2_h[0]);
    float2 inp2_h_f_1 = __half22float2(inp2_h[1]);

    float2 inp3_h_f_0 = __half22float2(inp3_h[0]);
    float2 inp3_h_f_1 = __half22float2(inp3_h[1]);

    inp1_h_f_0.x += (inp2_h_f_0.x + inp3_h_f_0.x);
    inp1_h_f_0.y += (inp2_h_f_0.y + inp3_h_f_0.y);
    inp1_h_f_1.x += (inp2_h_f_1.x + inp3_h_f_1.x);
    inp1_h_f_1.y += (inp2_h_f_1.y + inp3_h_f_1.y);

    float2 val_f;
    __half2* val_h = reinterpret_cast<__half2*>(&val_f);

    val_h[0] = __float22half2_rn(inp1_h_f_0);
    val_h[1] = __float22half2_rn(inp1_h_f_1);

    float2* out_4 = reinterpret_cast<float2*>(out);
    out_4[row * row_stride + id] = val_f;
}

template <>
void launch_fused_add3<float>(float* out,
                              const float* inp1,
                              const float* inp2,
                              const float* inp3,
                              int batch_size,
                              int seq_length,
                              int hidden_size,
                              cudaStream_t& stream)
{
    dim3 grid_dim(batch_size * seq_length);

    dim3 block_dim(hidden_size / 4);

    fused_add3_kernel<<<grid_dim, block_dim, 0, stream>>>(
        out, inp1, inp2, inp3, (batch_size * seq_length * hidden_size), hidden_size / 4);
}

template <>
void launch_fused_add3<__half>(__half* out,
                               const __half* inp1,
                               const __half* inp2,
                               const __half* inp3,
                               int batch_size,
                               int seq_length,
                               int hidden_size,
                               cudaStream_t& stream)
{
    dim3 grid_dim(batch_size * seq_length);

    dim3 block_dim(hidden_size / 4);

    fused_add3_kernel<<<grid_dim, block_dim, 0, stream>>>(
        out, inp1, inp2, inp3, (batch_size * seq_length * hidden_size), hidden_size / 4);
}

__global__ void fused_add4_kernel(float* out,
                                  const float* inp1,
                                  const float* inp2,
                                  const float* inp3,
                                  const float* inp4,
                                  int size,
                                  int row_stride)
{
    int row = blockIdx.x;
    int id = threadIdx.x;

    const float4* inp1_4 = reinterpret_cast<const float4*>(inp1);
    const float4* inp2_4 = reinterpret_cast<const float4*>(inp2);
    const float4* inp3_4 = reinterpret_cast<const float4*>(inp3);
    const float4* inp4_4 = reinterpret_cast<const float4*>(inp4);
    float4* out_4 = reinterpret_cast<float4*>(out);

    float4 val;
    float4 inp1_reg = inp1_4[row * row_stride + id];
    float4 inp2_reg = inp2_4[row * row_stride + id];
    float4 inp3_reg = inp3_4[row * row_stride + id];
    float4 inp4_reg = inp4_4[row * row_stride + id];

    val.x = inp1_reg.x + inp2_reg.x + inp3_reg.x + inp4_reg.x;
    val.y = inp1_reg.y + inp2_reg.y + inp3_reg.y + inp4_reg.y;
    val.z = inp1_reg.z + inp2_reg.z + inp3_reg.z + inp4_reg.z;
    val.w = inp1_reg.w + inp2_reg.w + inp3_reg.w + inp4_reg.w;

    out_4[row * row_stride + id] = val;
}

__global__ void fused_add4_kernel(__half* out,
                                  const __half* inp1,
                                  const __half* inp2,
                                  const __half* inp3,
                                  const __half* inp4,
                                  int size,
                                  int row_stride)
{
    int row = blockIdx.x;
    int id = threadIdx.x;
    const float2* inp1_arr = reinterpret_cast<const float2*>(inp1);
    const float2* inp2_arr = reinterpret_cast<const float2*>(inp2);
    const float2* inp3_arr = reinterpret_cast<const float2*>(inp3);
    const float2* inp4_arr = reinterpret_cast<const float2*>(inp4);

    float2 inp1_4 = inp1_arr[row * row_stride + id];
    float2 inp2_4 = inp2_arr[row * row_stride + id];
    float2 inp3_4 = inp3_arr[row * row_stride + id];
    float2 inp4_4 = inp4_arr[row * row_stride + id];

    __half2* inp1_h = reinterpret_cast<__half2*>(&inp1_4);
    __half2* inp2_h = reinterpret_cast<__half2*>(&inp2_4);
    __half2* inp3_h = reinterpret_cast<__half2*>(&inp3_4);
    __half2* inp4_h = reinterpret_cast<__half2*>(&inp4_4);

    float2 inp1_h_f_0 = __half22float2(inp1_h[0]);
    float2 inp1_h_f_1 = __half22float2(inp1_h[1]);

    float2 inp2_h_f_0 = __half22float2(inp2_h[0]);
    float2 inp2_h_f_1 = __half22float2(inp2_h[1]);

    float2 inp3_h_f_0 = __half22float2(inp3_h[0]);
    float2 inp3_h_f_1 = __half22float2(inp3_h[1]);

    float2 inp4_h_f_0 = __half22float2(inp4_h[0]);
    float2 inp4_h_f_1 = __half22float2(inp4_h[1]);

    inp1_h_f_0.x += (inp2_h_f_0.x + inp3_h_f_0.x + inp4_h_f_0.x);
    inp1_h_f_0.y += (inp2_h_f_0.y + inp3_h_f_0.y + inp4_h_f_0.y);
    inp1_h_f_1.x += (inp2_h_f_1.x + inp3_h_f_1.x + inp4_h_f_1.x);
    inp1_h_f_1.y += (inp2_h_f_1.y + inp3_h_f_1.y + inp4_h_f_1.y);

    float2 val_f;
    __half2* val_h = reinterpret_cast<__half2*>(&val_f);

    val_h[0] = __float22half2_rn(inp1_h_f_0);
    val_h[1] = __float22half2_rn(inp1_h_f_1);

    float2* out_4 = reinterpret_cast<float2*>(out);
    out_4[row * row_stride + id] = val_f;
}

template <>
void launch_fused_add4<float>(float* out,
                              const float* inp1,
                              const float* inp2,
                              const float* inp3,
                              const float* inp4,
                              int batch_size,
                              int seq_length,
                              int hidden_size,
                              cudaStream_t& stream)
{
    dim3 grid_dim(batch_size * seq_length);

    dim3 block_dim(hidden_size / 4);

    fused_add4_kernel<<<grid_dim, block_dim, 0, stream>>>(
        out, inp1, inp2, inp3, inp4, (batch_size * seq_length * hidden_size), hidden_size / 4);
}

template <>
void launch_fused_add4<__half>(__half* out,
                               const __half* inp1,
                               const __half* inp2,
                               const __half* inp3,
                               const __half* inp4,
                               int batch_size,
                               int seq_length,
                               int hidden_size,
                               cudaStream_t& stream)
{
    dim3 grid_dim(batch_size * seq_length);

    dim3 block_dim(hidden_size / 4);

    fused_add4_kernel<<<grid_dim, block_dim, 0, stream>>>(
        out, inp1, inp2, inp3, inp4, (batch_size * seq_length * hidden_size), hidden_size / 4);
}
