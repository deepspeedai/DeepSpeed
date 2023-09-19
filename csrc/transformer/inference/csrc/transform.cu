// Copyright (c) Microsoft Corporation.
// SPDX-License-Identifier: Apache-2.0

// DeepSpeed Team

#ifndef __HIP_PLATFORM_HCC__
#include <cuda_profiler_api.h>
#endif
#include "conversion_utils.h"
#include "inference_cuda_layers.h"
namespace cg = cooperative_groups;

// only used to avoid compilation error due to lack of definition.
#ifndef BF16_AVAILABLE
using __nv_bfloat162 = __half2;
#endif

// Bias add

__global__ void bias_add_transform_0213(float* output,
                                        float* k_cache,
                                        float* v_cache,
                                        const float* vals,
                                        const float* bias,
                                        int hidden_dim,
                                        int seq_length,
                                        unsigned seq_offset,
                                        int heads,
                                        int head_stride,
                                        int num_kv,
                                        int rotary_dim,
                                        bool rotate_half,
                                        bool rotate_every_two,
                                        int head_ext,
                                        int max_out_tokens)
{
    int d0_stride = hidden_dim * seq_length;
    int d1_stride = hidden_dim;
    int d2_stride = hidden_dim / heads;

    int d0 = blockIdx.x;                                                  // Batch
    int d1 = blockIdx.y;                                                  // Sequence ID (0-127)
    int cnt = blockIdx.z / head_ext;                                      // Hidden count
    int d2 = threadIdx.y + (blockIdx.z % head_ext) * (heads / head_ext);  // Head (0-11)
    int d3 = threadIdx.x;                                                 // Values (groups of 4)

    int d2_out_stride = d2_stride * (cnt == 0 ? seq_length : max_out_tokens);
    int d0_out_stride = hidden_dim * (cnt == 0 ? seq_length : max_out_tokens);

    const float4* vals_vec = reinterpret_cast<const float4*>(vals);
    float4* output_vec =
        reinterpret_cast<float4*>(cnt == 0 ? output : (cnt == 1 ? k_cache : v_cache));

    vals_vec += (d0 * (d1_stride + num_kv * 2 * d2_stride) * seq_length);
    vals_vec += d1 * (d1_stride + num_kv * 2 * d2_stride);
    vals_vec += (cnt == 0 ? 0 : d1_stride) + (cnt == 0 ? 0 : (cnt - 1) * num_kv * d2_stride);
    vals_vec += ((cnt == 0 ? d2 : (d2 / head_stride)) * d2_stride);

    output_vec += (d1 * d2_stride);
    output_vec += (d0 * d0_out_stride);
    output_vec += (d2 * d2_out_stride);

    unsigned seq_id = d1 + seq_offset;
    float4 inputs = vals_vec[d3];
    int lane = d3 & 0x1f;
    if (cnt < 2 && rotary_dim > 0 && d3 < rotary_dim) {
        float4 q = vals_vec[d3];
        float2* q_f = reinterpret_cast<float2*>(&q);
        if (rotate_every_two) {
#pragma unroll
            for (int o = 0; o < 2; o++) {
                float inv_freq = (float)(((d3 << 1) + o) * 2) / (float)(rotary_dim << 2);
                inv_freq = 1.0 / powf(10000.0, inv_freq) * (float)seq_id;
                q_f[o].x = (-1.0 * q_f[o].y * sinf(inv_freq) + q_f[o].x * cosf(inv_freq));
                q_f[o].y = (q_f[o].x * sinf(inv_freq) + q_f[o].y * cosf(inv_freq));
            }
        }
        output_vec[d3] = q;
    } else
        output_vec[d3] = inputs;
}

#define ATTN_H 3
#define MAX_SEQ_LINE 10

template <typename T>
__global__ void bias_add_transform_0213(T* output,  // q
                                        T* k_cache,
                                        T* v_cache,
                                        const T* vals,  // qkv
                                        const T* bias,
                                        int hidden_dim,
                                        int seq_length,
                                        unsigned seq_offset,
                                        int all_tokens,
                                        int heads,
                                        int head_stride,
                                        int num_kv,
                                        int rotary_dim,
                                        bool rotate_half,
                                        bool rotate_every_two,
                                        int head_ext,
                                        int max_out_tokens)
{
    using T2 =
        typename std::conditional<std::is_same<T, __half>::value, __half2, __nv_bfloat162>::type;
    unsigned half_dim = (rotary_dim << 3) >> 1;
    int d0_stride = hidden_dim * seq_length;
    int d1_stride = hidden_dim;
    int d2_stride = hidden_dim / heads;

    int d0 = blockIdx.x;                                                  // Batch
    int d1 = blockIdx.y;                                                  // Sequence ID (0-127)
    int cnt = blockIdx.z / head_ext;                                      // Hidden count
    int d2 = threadIdx.y + (blockIdx.z % head_ext) * (heads / head_ext);  // Head (0-11)
    int d3 = threadIdx.x;                                                 // Values (groups of 4)

    int d2_out_stride = d2_stride * (cnt == 0 ? seq_length : max_out_tokens);
    int d0_out_stride = hidden_dim * (cnt == 0 ? seq_length : max_out_tokens);

    float4 vals_arr;
    float4 output_arr;

    T2* vals_half = reinterpret_cast<T2*>(&vals_arr);
    T2* output_half = reinterpret_cast<T2*>(&output_arr);

    const float4* vals_vec = reinterpret_cast<const float4*>(vals);
    float4* output_vec =
        reinterpret_cast<float4*>(cnt == 0 ? output : (cnt == 1 ? k_cache : v_cache));

    vals_vec += (d0 * (d1_stride + num_kv * 2 * d2_stride) * seq_length);
    vals_vec += (d1 * (d1_stride + num_kv * 2 * d2_stride));
    vals_vec += (cnt == 0 ? 0 : d1_stride) + (cnt == 0 ? 0 : (cnt - 1) * num_kv * d2_stride);
    vals_vec += ((cnt == 0 ? d2 : (d2 / head_stride)) * d2_stride);

    output_vec += (d1 * d2_stride);
    output_vec += (d0 * d0_out_stride);
    output_vec += (d2 * d2_out_stride);

    unsigned seq_id = d1 + seq_offset;

    int lane = d3 & 0x1f;
    if (cnt < 2 && rotary_dim > 0 && d3 < rotary_dim) {
        float4 q = vals_vec[d3];
        T2* q_h = reinterpret_cast<T2*>(&q);
        if (rotate_every_two) {
#pragma unroll
            for (int o = 0; o < 4; o++) {
                float inv_freq = (float)(((d3 << 2) + o) * 2) / (float)(rotary_dim << 3);
                inv_freq = 1.0 / powf(10000.0, inv_freq) * (float)seq_id;
                float q_data[2];
                q_data[0] = conversion::to<float>(q_h[o].x);
                q_data[1] = conversion::to<float>(q_h[o].y);
                q_h[o].x = conversion::to<T>(-1.0 * q_data[1] * sinf(inv_freq) +
                                             q_data[0] * cosf(inv_freq));
                q_h[o].y =
                    conversion::to<T>(q_data[0] * sinf(inv_freq) + q_data[1] * cosf(inv_freq));
            }
        }
        output_vec[d3] = q;
    } else
        output_vec[d3] = vals_vec[d3];
}

// [B S C*H] - > C * [B A S N]
template <>
void launch_bias_add_transform_0213<float>(float* output,
                                           float* k_cache,
                                           float* v_cache,
                                           const float* vals,
                                           const float* bias,
                                           int batch_size,
                                           int seq_length,
                                           unsigned seq_offset,
                                           int all_tokens,
                                           int hidden_dim,
                                           int heads,
                                           int num_kv,
                                           int rotary_dim,
                                           bool rotate_half,
                                           bool rotate_every_two,
                                           cudaStream_t stream,
                                           int trans_count,
                                           int max_out_tokens)
{
    hidden_dim >>= 2;
    int head_ext = (hidden_dim - 1) / MAX_THREADS + 1;

    dim3 block_dim(hidden_dim / heads, (heads / head_ext));
    dim3 grid_dim(batch_size, seq_length, (trans_count * head_ext));

    bias_add_transform_0213<<<grid_dim, block_dim, 0, stream>>>(output,
                                                                k_cache,
                                                                v_cache,
                                                                vals,
                                                                bias,
                                                                hidden_dim,
                                                                seq_length,
                                                                seq_offset,
                                                                heads,
                                                                num_kv > 0 ? (heads / num_kv) : 1,
                                                                num_kv > 0 ? num_kv : heads,
                                                                rotary_dim >> 2,
                                                                rotate_half,
                                                                rotate_every_two,
                                                                head_ext,
                                                                max_out_tokens);
}

template <typename T>
void launch_bias_add_transform_0213(T* output,
                                    T* k_cache,
                                    T* v_cache,
                                    const T* vals,
                                    const T* bias,
                                    int batch_size,
                                    int seq_length,
                                    unsigned seq_offset,
                                    int all_tokens,
                                    int hidden_dim,
                                    int heads,
                                    int num_kv,
                                    int rotary_dim,
                                    bool rotate_half,
                                    bool rotate_every_two,
                                    cudaStream_t stream,
                                    int trans_count,
                                    int max_out_tokens)
{
    hidden_dim >>= 3;
    int head_ext = 1;  // (hidden_dim - 1) / MAX_THREADS + 1;
    dim3 block_dim(hidden_dim / heads, (heads / head_ext));
    dim3 grid_dim(batch_size, seq_length, (trans_count * head_ext));
    bias_add_transform_0213<<<grid_dim, block_dim, 0, stream>>>(output,
                                                                k_cache,
                                                                v_cache,
                                                                vals,
                                                                bias,
                                                                hidden_dim,
                                                                seq_length,
                                                                seq_offset,
                                                                all_tokens,
                                                                heads,
                                                                num_kv > 0 ? (heads / num_kv) : 1,
                                                                num_kv > 0 ? num_kv : heads,
                                                                rotary_dim >> 3,
                                                                rotate_half,
                                                                rotate_every_two,
                                                                head_ext,
                                                                max_out_tokens);
}

#define INSTANTIATE_LAUNCH_BIAS_ADD_TRANSFORM_0213(T)             \
    template void launch_bias_add_transform_0213<T>(T*,           \
                                                    T*,           \
                                                    T*,           \
                                                    const T*,     \
                                                    const T*,     \
                                                    int,          \
                                                    int,          \
                                                    unsigned,     \
                                                    int,          \
                                                    int,          \
                                                    int,          \
                                                    int,          \
                                                    int,          \
                                                    bool,         \
                                                    bool,         \
                                                    cudaStream_t, \
                                                    int,          \
                                                    int)

#ifdef BF16_AVAILABLE
INSTANTIATE_LAUNCH_BIAS_ADD_TRANSFORM_0213(__nv_bfloat16);
#endif
INSTANTIATE_LAUNCH_BIAS_ADD_TRANSFORM_0213(__half);

// Bias add

__global__ void pad_add_transform_0213(float* output,
                                       const float* vals,
                                       int hidden_dim,
                                       int seq_length,
                                       int padded_seq_len,
                                       int heads,
                                       int padded_head_size)
{
}

template <typename T>
__global__ void pad_add_transform_0213(T* output,
                                       const T* vals,
                                       int hidden_dim,
                                       int seq_length,
                                       int padded_seq_len,
                                       int heads,
                                       int padded_head_size)
{
    using T2 =
        typename std::conditional<std::is_same<T, __half>::value, __half2, __nv_bfloat162>::type;
    float4 ZERO;
    const T2 zero_h = conversion::to<T2>(0.f);
    T2* ZERO_h = reinterpret_cast<T2*>(&ZERO);
#pragma unroll
    for (int i = 0; i < 4; i++) ZERO_h[i] = zero_h;

    int d0_stride = hidden_dim * seq_length;
    int d1_stride = hidden_dim;
    int d2_stride = hidden_dim / heads;

    int d0 = blockIdx.x;                             // Batch
    int d1 = blockIdx.y * blockDim.z + threadIdx.z;  // Sequence ID (0-127)
    int d2 = threadIdx.y;                            // Head (0-11)
    int d3 = threadIdx.x;                            // Values (groups of 4)

    int d2_out_stride = padded_head_size * padded_seq_len;
    int d0_out_stride = heads * d2_out_stride;

    const float4* vals_vec = reinterpret_cast<const float4*>(vals);
    float4* output_vec = reinterpret_cast<float4*>(output);

    vals_vec += (d0 * d0_stride);
    vals_vec += (d1 * d1_stride);
    vals_vec += (d2 * d2_stride);

    output_vec += (d1 * padded_head_size);
    output_vec += (d0 * d0_out_stride);
    output_vec += (d2 * d2_out_stride);

    if (d3 < d2_stride && d1 < seq_length)
        output_vec[d3] = vals_vec[d3];
    else
        output_vec[d3] = ZERO;
}

// [B S C*H] - > C * [B A S N]
template <>
void launch_pad_add_transform_0213<float>(float* output,
                                          const float* vals,
                                          int batch_size,
                                          int hidden_dim,
                                          int seq_length,
                                          int padded_seq_len,
                                          int heads,
                                          int padded_head_size,
                                          cudaStream_t stream)
{
}

template <typename T>
void launch_pad_add_transform_0213(T* output,
                                   const T* vals,
                                   int batch_size,
                                   int hidden_dim,
                                   int seq_length,
                                   int padded_seq_len,
                                   int heads,
                                   int padded_head_size,
                                   cudaStream_t stream)
{
    hidden_dim >>= 3;
    dim3 block_dim((padded_head_size >> 3), heads, 2);
    dim3 grid_dim(batch_size, padded_seq_len / 2);
    pad_add_transform_0213<<<grid_dim, block_dim, 0, stream>>>(
        output, vals, hidden_dim, seq_length, padded_seq_len, heads, padded_head_size >> 3);
}

#define INSTANTIATE_LAUNCH_PAD_ADD_TRANSFORM_0213_SIMPLE(T) \
    template void launch_pad_add_transform_0213<T>(         \
        T*, const T*, int, int, int, int, int, int, cudaStream_t);

INSTANTIATE_LAUNCH_PAD_ADD_TRANSFORM_0213_SIMPLE(__half);
#ifdef BF16_AVAILABLE
INSTANTIATE_LAUNCH_PAD_ADD_TRANSFORM_0213_SIMPLE(__nv_bfloat16);
#endif

// Bias add
template <typename T>
__global__ void bias_add_transform_0213(T* output,
                                        const T* vals,
                                        const T* bias,
                                        int hidden_dim,
                                        int seq_length,
                                        int heads,
                                        int head_ext);

template <>
__global__ void bias_add_transform_0213<float>(float* output,
                                               const float* vals,
                                               const float* bias,
                                               int hidden_dim,
                                               int seq_length,
                                               int heads,
                                               int head_ext)
{
    int d0_stride = hidden_dim * seq_length;
    int d1_stride = hidden_dim;
    int d2_stride = hidden_dim / heads;

    int d0_out_stride = d0_stride;
    int d1_out_stride = d2_stride;
    int d2_out_stride = d2_stride * seq_length;

    int d0 = blockIdx.x;                                                  // Batch
    int d1 = blockIdx.y;                                                  // Sequence ID (0-127)
    int cnt = blockIdx.z / head_ext;                                      // Hidden count
    int d2 = threadIdx.y + (blockIdx.z % head_ext) * (heads / head_ext);  // Head (0-11)
    int d3 = threadIdx.x;                                                 // Values (groups of 4)

    const float4* vals_vec = reinterpret_cast<const float4*>(vals);
    const float4* bias_vec = reinterpret_cast<const float4*>(bias);
    float4* output_vec = reinterpret_cast<float4*>(output);

    float4 inputs = vals_vec[d0 * d0_stride * (gridDim.z / head_ext) + cnt * d1_stride +
                             d1 * d1_stride * (gridDim.z / head_ext) + d2 * d2_stride + d3];
    float4 biases = bias_vec[cnt * d1_stride + d2 * d2_stride + d3];

    float4 outputs;
    outputs.x = inputs.x + biases.x;
    outputs.y = inputs.y + biases.y;
    outputs.z = inputs.z + biases.z;
    outputs.w = inputs.w + biases.w;

    output_vec[cnt * d0_out_stride * gridDim.x + d0 * d0_out_stride + d1 * d1_out_stride +
               d2 * d2_out_stride + d3] = outputs;
}

template <typename T>
__global__ void bias_add_transform_0213(T* output,
                                        const T* vals,
                                        const T* bias,
                                        int hidden_dim,
                                        int seq_length,
                                        int heads,
                                        int head_ext)
{
    using T2 =
        typename std::conditional<std::is_same<T, __half>::value, __half2, __nv_bfloat162>::type;
    int d0_stride = hidden_dim * seq_length;
    int d1_stride = hidden_dim;
    int d2_stride = hidden_dim / heads;

    int d2_out_stride = d2_stride * seq_length;

    int d0 = blockIdx.x;                                                  // Batch
    int d1 = blockIdx.y;                                                  // Sequence ID (0-127)
    int cnt = blockIdx.z / head_ext;                                      // Hidden count
    int d2 = threadIdx.y + (blockIdx.z % head_ext) * (heads / head_ext);  // Head (0-11)
    int d3 = threadIdx.x;                                                 // Values (groups of 4)

    float4 vals_arr;
    float4 bias_arr;
    float4 output_arr;
    T2* vals_half = reinterpret_cast<T2*>(&vals_arr);
    T2* bias_half = reinterpret_cast<T2*>(&bias_arr);
    T2* output_half = reinterpret_cast<T2*>(&output_arr);

    const float4* vals_vec = reinterpret_cast<const float4*>(vals);
    const float4* bias_vec = reinterpret_cast<const float4*>(bias);
    float4* output_vec = reinterpret_cast<float4*>(output);

    vals_vec += (d0 * d0_stride * (gridDim.z / head_ext));
    vals_vec += (d1 * d1_stride * (gridDim.z / head_ext));
    vals_vec += (cnt * d1_stride);
    vals_vec += (d2 * d2_stride);

    bias_vec += (cnt * d1_stride);
    bias_vec += (d2 * d2_stride);

    output_vec += (cnt * d0_stride * gridDim.x);
    output_vec += (d1 * d2_stride);
    output_vec += (d0 * d0_stride);
    output_vec += (d2 * d2_out_stride);

    bias_arr = bias_vec[d3];
    vals_arr = vals_vec[d3];

    output_half[0] = vals_half[0] + bias_half[0];
    output_half[1] = vals_half[1] + bias_half[1];
    output_half[2] = vals_half[2] + bias_half[2];
    output_half[3] = vals_half[3] + bias_half[3];
    output_vec[d3] = output_arr;
}

template <typename T>
__global__ void bias_add_transform_0213_v2(T* output,
                                           const T* vals,
                                           const T* bias,
                                           int hidden_dim,
                                           int seq_length,
                                           int heads)
{
    using T2 =
        typename std::conditional<std::is_same<T, __half>::value, __half2, __nv_bfloat162>::type;
    __shared__ float4 in_data[3072];

    int d0_stride = hidden_dim * seq_length;
    int d1_stride = hidden_dim;
    int d2_stride = hidden_dim / heads;
    int iteration_stride = d1_stride * blockDim.z;  // Hidden * 3 / 8
    int batch_stride = d0_stride * blockDim.z;      // Hidden * S * 3 / 8

    int d0_out_stride = d0_stride;
    int d1_out_stride = d2_stride;
    int d2_out_stride = d2_stride * seq_length;

    int d0 = blockIdx.x;    // Batch
    int d1 = blockIdx.y;    // Sequence ID (0-127)
    int cnt = threadIdx.z;  // blockIdx.z; // Hidden count
    int d2 = threadIdx.y;   // Head (0-11)
    int d3 = threadIdx.x;   // Values (groups of 4)

    float4 vals_arr[1];
    float4 bias_arr[1];
    float4 output_arr[1];
    T2* vals_half = reinterpret_cast<T2*>(vals_arr);
    T2* bias_half = reinterpret_cast<T2*>(bias_arr);
    T2* output_half = reinterpret_cast<T2*>(output_arr);

    const float4* vals_vec = reinterpret_cast<const float4*>(vals);
    const float4* bias_vec = reinterpret_cast<const float4*>(bias);
    float4* output_vec = reinterpret_cast<float4*>(output);

    int iter_index = cnt * d1_stride + d2 * d2_stride + d3;
    int input_offset = d0 * batch_stride + d1 * (iteration_stride << 1);
    bias_arr[0] = bias_vec[iter_index];

#pragma unroll
    for (int iter = 0; iter < 2; iter++) {
        int iter_id = iter * iteration_stride + iter_index;
        vals_arr[0] = vals_vec[input_offset + iter_id];

        output_half[0] = vals_half[0] + bias_half[0];
        output_half[1] = vals_half[1] + bias_half[1];
        output_half[2] = vals_half[2] + bias_half[2];
        output_half[3] = vals_half[3] + bias_half[3];

        in_data[iter_id] = output_arr[0];
    }
    __syncthreads();

    iteration_stride = blockDim.z * (blockDim.y >> 1);
    int matrix_stride = (d0_out_stride * gridDim.x);
    int head_count = (d2 >> 1) + cnt * (blockDim.y >> 1);

    int out_index = d0 * d0_out_stride + d1 * (d1_out_stride << 1) + d3 + (d2 % 2) * d2_stride;

#pragma unroll
    for (int iter = 0; iter < 2; iter++) {
        int iter_row = (iter * iteration_stride) + head_count;
        int iter_offset =
            (iter_row % blockDim.y) * d2_out_stride + (iter_row / blockDim.y) * matrix_stride;
        output_vec[out_index + iter_offset] =
            in_data[iter_row * d2_stride + d3 + (d2 % 2) * (d1_stride * blockDim.z)];
    }
}

template <typename T>
__global__ void transform4d_0213(T* out,
                                 const T* in,
                                 int heads,
                                 int seq_length,
                                 int hidden_dim,
                                 int head_ext);

template <>
__global__ void transform4d_0213<float>(float* out,
                                        const float* in,
                                        int heads,
                                        int seq_length,
                                        int hidden_dim,
                                        int head_ext)
{
    int d0_stride = hidden_dim * seq_length;
    int d1_stride = d0_stride / heads;
    int d2_stride = hidden_dim / heads;

    int d0_out_stride = d0_stride;
    int d1_out_stride = d2_stride;
    int d2_out_stride = hidden_dim;

    int d0 = blockIdx.x;                                        // Batch
    int d1 = blockIdx.y / ((seq_length - 1) / blockDim.y + 1);  // Head
    int d2 = (threadIdx.y + blockDim.y * blockIdx.y) % seq_length;
    int cnt = blockIdx.z;
    int d3 = threadIdx.x;  // Values (groups of 8)

    if (d2 < seq_length) {
        const float4* in_vec = reinterpret_cast<const float4*>(in);
        float4* out_vec = reinterpret_cast<float4*>(out);

        float4 vals_vec = in_vec[cnt * d0_stride * gridDim.x + d0 * d0_stride + d1 * d1_stride +
                                 d2 * d2_stride + d3];
        out_vec[d0 * d0_out_stride * gridDim.z + cnt * d2_out_stride + d1 * d1_out_stride +
                d2 * d2_out_stride * gridDim.z + d3] = vals_vec;
    }
}

template <typename T>
__global__ void transform4d_0213(T* out,
                                 const T* in,
                                 int heads,
                                 int seq_length,
                                 int hidden_dim,
                                 int head_ext)
{
    int d0_stride = hidden_dim * (seq_length / head_ext);
    int d1_stride = hidden_dim;
    int d2_stride = hidden_dim / heads;

    int d0 = blockIdx.x;                                                  // Batch
    int d1 = threadIdx.y + (blockIdx.z % head_ext) * (heads / head_ext);  // Head
    int d2 = blockIdx.z / head_ext;                                       // Sequence
    int cnt = blockIdx.y;                                                 // Hidden count
    int d3 = threadIdx.x;                                                 // Values (groups of 8)

    const float4* in_vec = reinterpret_cast<const float4*>(in);
    float4* out_vec = reinterpret_cast<float4*>(out);

    in_vec += (cnt * d0_stride * gridDim.x);
    in_vec += (d0 * d0_stride);
    in_vec += (d2 * d2_stride);
    in_vec += (d1 * d2_stride * seq_length);

    out_vec += (cnt * d1_stride);
    out_vec += (d1 * d2_stride);
    out_vec += (d0 * d0_stride * gridDim.y);
    out_vec += (d2 * d1_stride * gridDim.y);

    out_vec[d3] = in_vec[d3];
}

template <typename T>
__global__ void transform4d_0213_v2(T* out, const T* in, int heads, int seq_length, int hidden_dim)
{
    __shared__ float4 in_data[3072];

    int d0_stride = hidden_dim * seq_length;
    int d1_stride = hidden_dim;
    int d2_stride = hidden_dim / heads;

    int d0 = blockIdx.x;    // Batch
    int d1 = threadIdx.y;   // Head
    int d2 = blockIdx.y;    // Sequence
    int cnt = threadIdx.z;  // Hidden count
    int d3 = threadIdx.x;   // Values (groups of 8)

    const float4* in_vec = reinterpret_cast<const float4*>(in);
    float4* out_vec = reinterpret_cast<float4*>(out);

    int input_offset = d0 * d0_stride + d2 * (d2_stride << 1) + d3 + (d1 % 2) * d2_stride;
    int head_count = (d1 >> 1) + cnt * (blockDim.y >> 1);
    int iteration_stride = blockDim.z * (blockDim.y >> 1);
    int matrix_stride = (d0_stride * gridDim.x);

#pragma unroll
    for (int iter = 0; iter < 2; iter++) {
        int iter_row = iter * iteration_stride + head_count;
        int iter_offset = (iter_row % blockDim.y) * d2_stride;

        in_data[d3 + iter_offset + (iter_row / blockDim.y + (d1 % 2) * blockDim.z) * d1_stride] =
            in_vec[input_offset + iter_offset * seq_length +
                   (iter_row / blockDim.y) * matrix_stride];
    }
    __syncthreads();

    iteration_stride = d1_stride * blockDim.z;
    int iter_index = cnt * d1_stride + d1 * d2_stride + d3;
    int output_offset = d0 * d0_stride * blockDim.z + d2 * (iteration_stride << 1);

#pragma unroll
    for (int iter = 0; iter < 2; iter++) {
        int iter_id = iter * iteration_stride + iter_index;
        out_vec[output_offset + iter_id] = in_data[iter_id];
    }
}

// 3 * [B A S N] - > [B S C*H]
template <>
void launch_transform4d_0213<float>(float* out,
                                    const float* in,
                                    int batch_size,
                                    int heads,
                                    int seq_length,
                                    int hidden_dim,
                                    cudaStream_t stream,
                                    int trans_count)
{
    hidden_dim >>= 2;
    dim3 grid_dims(batch_size, heads * ((seq_length - 1) / 8 + 1), trans_count);
    dim3 block_dims(hidden_dim / heads, 8);
    transform4d_0213<float>
        <<<grid_dims, block_dims, 0, stream>>>(out, in, heads, seq_length, hidden_dim, 1);
}

template <typename T>
void launch_transform4d_0213(T* out,
                             const T* in,
                             int batch_size,
                             int heads,
                             int seq_length,
                             int hidden_dim,
                             cudaStream_t stream,
                             int trans_count)
{
    hidden_dim >>= 3;
    int head_ext = (hidden_dim - 1) / MAX_THREADS + 1;
    dim3 grid_dims(batch_size, trans_count, (seq_length * head_ext));
    dim3 block_dims(hidden_dim / heads, (heads / head_ext));
    transform4d_0213<<<grid_dims, block_dims, 0, stream>>>(
        out, in, heads, seq_length, hidden_dim, head_ext);
}

#define INSTANTIATE_2B_LAUNCH_TRANSFORM4D(T) \
    template void launch_transform4d_0213<T>(T*, const T*, int, int, int, int, cudaStream_t, int);

INSTANTIATE_2B_LAUNCH_TRANSFORM4D(__half)
#ifdef BF16_AVAILABLE
INSTANTIATE_2B_LAUNCH_TRANSFORM4D(__nv_bfloat16)
#endif
