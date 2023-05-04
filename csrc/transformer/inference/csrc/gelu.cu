// Copyright (c) Microsoft Corporation.
// SPDX-License-Identifier: Apache-2.0

// DeepSpeed Team

#include "conversion_utils.h"
#include "inference_cuda_layers.h"
#include "memory_access_utils.h"

namespace cg = cooperative_groups;
#define MAX_CAP 4
#define MAX_SEQ 2048

// only used to avoid compilation error due to lack of definition.
#ifndef BF16_AVAILABLE
using __nv_bfloat162 = __half2;
#endif

inline __device__ float gelu(const float x)
{
    const float sqrt_param = 0.79788456080286535587989211986876f;
    const float mul_param = 0.044715;
    return x * 0.5f * (1.0f + tanhf(sqrt_param * (x + mul_param * x * x * x)));
}

/*
In-place gelu(biasAdd(x)) for channels last
*/
template <typename T>
__global__ void fused_bias_gelu(T* input, const T* bias, int total_count, int intermediate_size)
{
    // Input restriction: intermediate_size % vals_per_access == 0
    constexpr int granularity = 16;
    constexpr int values_per_access = granularity / sizeof(T);
    const int offset = (blockIdx.x * blockDim.x + threadIdx.x) * values_per_access;

    if (offset < total_count) {
        T data[values_per_access];
        T data_bias[values_per_access];
        mem_access::load_global<granularity>(data, input + offset);
        mem_access::load_global<granularity>(
            data_bias, bias + (offset % intermediate_size), bias != nullptr);

#pragma unroll
        for (int i = 0; i < values_per_access; i++) {
            float data_f = conversion::to<float>(data[i]);
            float bias_f = conversion::to<float>(data_bias[i]);
            data[i] = conversion::to<T>(gelu(data_f + bias_f));
        }

        mem_access::store_global<granularity>(input + offset, data);
    }
}

template <typename T>
void launch_bias_gelu(T* input,
                      const T* bias,
                      int intermediate_size,
                      int batch_size,
                      cudaStream_t stream)
{
    constexpr int threads = 1024;
    constexpr int granularity = 16;

    const int total_count = batch_size * intermediate_size;
    const int elems_per_block = threads * (granularity / sizeof(T));
    dim3 block_dims(threads);
    dim3 grid_dims((total_count + elems_per_block - 1) / elems_per_block);

    fused_bias_gelu<<<grid_dims, block_dims, 0, stream>>>(
        input, bias, total_count, intermediate_size);
}

#define INSTANTIATE_LAUNCH_BIAS_GELU(T) \
    template void launch_bias_gelu<T>(T*, const T*, int, int, cudaStream_t);

INSTANTIATE_LAUNCH_BIAS_GELU(float)
#ifdef BF16_AVAILABLE
INSTANTIATE_LAUNCH_BIAS_GELU(__nv_bfloat16)
#endif
INSTANTIATE_LAUNCH_BIAS_GELU(__half)

/*
In-place channels-last bias add
*/
template <typename T>
__global__ void fused_bias_add(T* input, const T* bias, int total_count, int intermediate_size)
{
    // Input restriction: intermediate_size % vals_per_access == 0
    constexpr int granularity = 16;
    constexpr int values_per_access = granularity / sizeof(T);
    const int offset = (blockIdx.x * blockDim.x + threadIdx.x) * values_per_access;

    if (offset < total_count) {
        T data[values_per_access];
        T data_bias[values_per_access];
        mem_access::load_global<granularity>(data, input + offset);
        mem_access::load_global<granularity>(
            data_bias, bias + (offset % intermediate_size), bias != nullptr);

#pragma unroll
        for (int i = 0; i < values_per_access; i++) {
            float data_f = conversion::to<float>(data[i]);
            float bias_f = conversion::to<float>(data_bias[i]);
            data[i] = conversion::to<T>(data_f + bias_f);
        }

        mem_access::store_global<granularity>(input + offset, data);
    }
}

template <typename T>
void launch_bias_add(T* input,
                     const T* bias,
                     int intermediate_size,
                     int batch_size,
                     cudaStream_t stream)
{
    constexpr int threads = 1024;
    constexpr int granularity = 16;

    const int total_count = batch_size * intermediate_size;
    const int elems_per_block = threads * (granularity / sizeof(T));
    dim3 block_dims(threads);
    dim3 grid_dims((total_count + elems_per_block - 1) / elems_per_block);

    fused_bias_add<<<grid_dims, block_dims, 0, stream>>>(
        input, bias, total_count, intermediate_size);
}

#define INSTANTIATE_LAUNCH_BIAS_ADD(T) \
    template void launch_bias_add<T>(T*, const T*, int, int, cudaStream_t);

INSTANTIATE_LAUNCH_BIAS_ADD(float)
#ifdef BF16_AVAILABLE
INSTANTIATE_LAUNCH_BIAS_ADD(__nv_bfloat16)
#endif
INSTANTIATE_LAUNCH_BIAS_ADD(__half)

__global__ void fused_bias_residual(float* residual,
                                    const float* hidden_state,
                                    const float* attn,
                                    const float* bias,
                                    const float* attn_bias,
                                    const int total_count,
                                    const int intermediate_size,
                                    const float mp_scale,
                                    const bool preln)
{
    float4* res_fl4_ptr = reinterpret_cast<float4*>(residual);
    const float4* hs_fl4_ptr = reinterpret_cast<const float4*>(hidden_state);
    const float4* attn_fl4_ptr = reinterpret_cast<const float4*>(attn);
    const float4* bias_fl4_ptr = reinterpret_cast<const float4*>(bias);
    const float4* attn_bias_fl4_ptr = reinterpret_cast<const float4*>(attn_bias);
    const int offset = blockIdx.x * blockDim.x + threadIdx.x;

    if (offset < total_count) {
        float4 res_fl4 = res_fl4_ptr[offset];
        const float4 hs_fl4 = hs_fl4_ptr[offset];
        const float4 attn_fl4 = attn_fl4_ptr[offset];
        const float4 bias_fl4 = bias_fl4_ptr[offset % intermediate_size];
        const float4 attn_bias_fl4 = attn_bias_fl4_ptr[offset % intermediate_size];
        if (preln) {
            // residual = (residual + attention + bias + attention_bias) *
            // mp_scale + hidden_state
            res_fl4.x =
                (res_fl4.x + attn_fl4.x + bias_fl4.x + attn_bias_fl4.x) * mp_scale + (hs_fl4.x);
            res_fl4.y =
                (res_fl4.y + attn_fl4.y + bias_fl4.y + attn_bias_fl4.y) * mp_scale + (hs_fl4.y);
            res_fl4.z =
                (res_fl4.z + attn_fl4.z + bias_fl4.z + attn_bias_fl4.z) * mp_scale + (hs_fl4.z);
            res_fl4.w =
                (res_fl4.w + attn_fl4.w + bias_fl4.w + attn_bias_fl4.w) * mp_scale + (hs_fl4.w);
        } else {
            // residual += hidden_state + bias
            res_fl4.x = res_fl4.x + hs_fl4.x + bias_fl4.x;
            res_fl4.y = res_fl4.y + hs_fl4.y + bias_fl4.y;
            res_fl4.z = res_fl4.z + hs_fl4.z + bias_fl4.z;
            res_fl4.w = res_fl4.w + hs_fl4.w + bias_fl4.w;
        }
        res_fl4_ptr[offset] = res_fl4;
    }
}

template <typename T>
__global__ void fused_bias_residual(T* residual,
                                    const T* hidden_state,
                                    const T* attn,
                                    const T* bias,
                                    const T* attn_bias,
                                    const int total_count,
                                    const int intermediate_size,
                                    const float mp_scale,
                                    const bool preln)
{
    using T2 =
        typename std::conditional<std::is_same<T, __half>::value, __half2, __nv_bfloat162>::type;
    float2* res_fl2_ptr = reinterpret_cast<float2*>(residual);
    const float2* hs_fl2_ptr = reinterpret_cast<const float2*>(hidden_state);
    const float2* attn_fl2_ptr = reinterpret_cast<const float2*>(attn);
    const float2* bias_fl2_ptr = reinterpret_cast<const float2*>(bias);
    const float2* attn_bias_fl2_ptr = reinterpret_cast<const float2*>(attn_bias);
    const int offset = blockIdx.x * blockDim.x + threadIdx.x;

    if (offset < total_count) {
        float2 res_fl2 = res_fl2_ptr[offset];
        const float2 hs_fl2 = hs_fl2_ptr[offset];
        const float2 attn_fl2 = attn_fl2_ptr[offset];
        const float2 bias_fl2 = bias_fl2_ptr[offset % intermediate_size];
        const float2 attn_bias_fl2 = attn_bias_fl2_ptr[offset % intermediate_size];

        T2* res_half2 = reinterpret_cast<T2*>(&res_fl2);
        const T2* hs_half2 = reinterpret_cast<const T2*>(&hs_fl2);
        const T2* attn_half2 = reinterpret_cast<const T2*>(&attn_fl2);
        const T2* bias_half2 = reinterpret_cast<const T2*>(&bias_fl2);
        const T2* attn_bias_half2 = reinterpret_cast<const T2*>(&attn_bias_fl2);

        float2 res_low = conversion::to<float2>(res_half2[0]);
        float2 res_high = conversion::to<float2>(res_half2[1]);

        const float2 hs_low = conversion::to<float2>(hs_half2[0]);
        const float2 hs_high = conversion::to<float2>(hs_half2[1]);

        const float2 attn_low = conversion::to<float2>(attn_half2[0]);
        const float2 attn_high = conversion::to<float2>(attn_half2[1]);

        const float2 bias_low = conversion::to<float2>(bias_half2[0]);
        const float2 bias_high = conversion::to<float2>(bias_half2[1]);

        const float2 attn_bias_low = conversion::to<float2>(attn_bias_half2[0]);
        const float2 attn_bias_high = conversion::to<float2>(attn_bias_half2[1]);

        if (preln) {
            // residual = (residual + attention + bias + attention_bias) *
            // mp_scale + hidden_state
            res_low.x =
                (res_low.x + attn_low.x + bias_low.x + attn_bias_low.x) * mp_scale + hs_low.x;
            res_low.y =
                (res_low.y + attn_low.y + bias_low.y + attn_bias_low.y) * mp_scale + hs_low.y;
            res_high.x =
                (res_high.x + attn_high.x + bias_high.x + attn_bias_high.x) * mp_scale + hs_high.x;
            res_high.y =
                (res_high.y + attn_high.y + bias_high.y + attn_bias_high.y) * mp_scale + hs_high.y;
        } else {
            // residual += hidden_state + bias
            res_low.x = (res_low.x + hs_low.x + bias_low.x);
            res_low.y = (res_low.y + hs_low.y + bias_low.y);
            res_high.x = (res_high.x + hs_high.x + bias_high.x);
            res_high.y = (res_high.y + hs_high.y + bias_high.y);
        }
        res_half2[0] = conversion::to<T2>(res_low);
        res_half2[1] = conversion::to<T2>(res_high);

        res_fl2_ptr[offset] = res_fl2;
    }
}

template <typename T>
void launch_bias_residual(T* residual,
                          T* hidden_state,
                          T* attn,
                          T* bias,
                          T* attn_bias,
                          int batch,
                          int hidden_dim,
                          int mp_size,
                          bool preln,
                          cudaStream_t stream)
{
    int total_count = batch * hidden_dim / 4;
    dim3 block_dims(1024);
    dim3 grid_dims((total_count - 1) / 1024 + 1);  // (batch_size);

    fused_bias_residual<<<grid_dims, block_dims, 0, stream>>>(residual,
                                                              hidden_state,
                                                              attn,
                                                              bias,
                                                              attn_bias,
                                                              total_count,
                                                              hidden_dim / 4,
                                                              1.0 / mp_size,
                                                              preln);
}

#define INSTANTIATE_LAUNCH_BIAS_RESIDUAL(T) \
    template void launch_bias_residual<T>(T*, T*, T*, T*, T*, int, int, int, bool, cudaStream_t);

INSTANTIATE_LAUNCH_BIAS_RESIDUAL(float);
#ifdef BF16_AVAILABLE
INSTANTIATE_LAUNCH_BIAS_RESIDUAL(__nv_bfloat16);
#endif
INSTANTIATE_LAUNCH_BIAS_RESIDUAL(__half);

__global__ void gptj_residual_add(float* residual,
                                  const float* hidden_state,
                                  const float* attn,
                                  const float* bias,
                                  const float* attn_bias,
                                  const int total_count,
                                  const int intermediate_size,
                                  const float mp_scale)
{
    float4* res_fl4_ptr = reinterpret_cast<float4*>(residual);
    const float4* hs_fl4_ptr = reinterpret_cast<const float4*>(hidden_state);
    const float4* attn_fl4_ptr = reinterpret_cast<const float4*>(attn);
    const float4* bias_fl4_ptr = reinterpret_cast<const float4*>(bias);
    const float4* attn_bias_fl4_ptr = reinterpret_cast<const float4*>(attn_bias);
    const int offset = blockIdx.x * blockDim.x + threadIdx.x;

    if (offset < total_count) {
        float4 res_fl4 = res_fl4_ptr[offset];
        const float4 hs_fl4 = hs_fl4_ptr[offset];
        const float4 attn_fl4 = attn_fl4_ptr[offset];
        const float4 bias_fl4 = bias_fl4_ptr[offset % intermediate_size];

        if (attn_bias) {
            float4 attn_bias_fl4 = attn_bias_fl4_ptr[offset % intermediate_size];
            // residual += attention_bias
            res_fl4.x += attn_bias_fl4.x;
            res_fl4.y += attn_bias_fl4.y;
            res_fl4.z += attn_bias_fl4.z;
            res_fl4.w += attn_bias_fl4.w;
        }
        // residual = hidden_state + attention + (residual + bias) * mp_scale
        res_fl4.x = hs_fl4.x + attn_fl4.x + (res_fl4.x + bias_fl4.x) * mp_scale;
        res_fl4.y = hs_fl4.y + attn_fl4.y + (res_fl4.y + bias_fl4.y) * mp_scale;
        res_fl4.z = hs_fl4.z + attn_fl4.z + (res_fl4.z + bias_fl4.z) * mp_scale;
        res_fl4.w = hs_fl4.w + attn_fl4.w + (res_fl4.w + bias_fl4.w) * mp_scale;

        res_fl4_ptr[offset] = res_fl4;
    }
}

template <typename T>
__global__ void gptj_residual_add(T* residual,
                                  const T* hidden_state,
                                  const T* attn,
                                  const T* bias,
                                  const T* attn_bias,
                                  const int total_count,
                                  const int intermediate_size,
                                  const float mp_scale)
{
    using T2 =
        typename std::conditional<std::is_same<T, __half>::value, __half2, __nv_bfloat162>::type;
    float2* res_fl2_ptr = reinterpret_cast<float2*>(residual);
    const float2* hs_fl2_ptr = reinterpret_cast<const float2*>(hidden_state);
    const float2* attn_fl2_ptr = reinterpret_cast<const float2*>(attn);
    const float2* bias_fl2_ptr = reinterpret_cast<const float2*>(bias);
    const float2* attn_bias_fl2_ptr = reinterpret_cast<const float2*>(attn_bias);
    const int offset = blockIdx.x * blockDim.x + threadIdx.x;

    if (offset < total_count) {
        float2 res_fl2 = res_fl2_ptr[offset];
        const float2 hs_fl2 = hs_fl2_ptr[offset];
        const float2 attn_fl2 = attn_fl2_ptr[offset];
        const float2 bias_fl2 = bias_fl2_ptr[offset % intermediate_size];

        T2* res_half2 = reinterpret_cast<T2*>(&res_fl2);
        const T2* hs_half2 = reinterpret_cast<const T2*>(&hs_fl2);
        const T2* attn_half2 = reinterpret_cast<const T2*>(&attn_fl2);
        const T2* bias_half2 = reinterpret_cast<const T2*>(&bias_fl2);

        float2 res_low = conversion::to<float2>(res_half2[0]);
        float2 res_high = conversion::to<float2>(res_half2[1]);

        const float2 hs_low = conversion::to<float2>(hs_half2[0]);
        const float2 hs_high = conversion::to<float2>(hs_half2[1]);

        const float2 attn_low = conversion::to<float2>(attn_half2[0]);
        const float2 attn_high = conversion::to<float2>(attn_half2[1]);

        const float2 bias_low = conversion::to<float2>(bias_half2[0]);
        const float2 bias_high = conversion::to<float2>(bias_half2[1]);

        if (attn_bias) {
            const float2 attn_bias_fl2 = attn_bias_fl2_ptr[offset % intermediate_size];
            const T2* attn_bias_half2 = reinterpret_cast<const T2*>(&attn_bias_fl2);
            const float2 attn_bias_low = conversion::to<float2>(attn_bias_half2[0]);
            const float2 attn_bias_high = conversion::to<float2>(attn_bias_half2[1]);
            // residual += attention_bias
            res_low.x += attn_bias_low.x;
            res_low.y += attn_bias_low.y;
            res_high.x += attn_bias_high.x;
            res_high.y += attn_bias_high.y;
        }
        // residual = hidden_state + attention + (residual + bias) * mp_scale
        res_low.x = attn_low.x + hs_low.x + (res_low.x + bias_low.x) * mp_scale;
        res_low.y = attn_low.y + hs_low.y + (res_low.y + bias_low.y) * mp_scale;
        res_high.x = attn_high.x + hs_high.x + (res_high.x + bias_high.x) * mp_scale;
        res_high.y = attn_high.y + hs_high.y + (res_high.y + bias_high.y) * mp_scale;

        res_half2[0] = conversion::to<T2>(res_low);
        res_half2[1] = conversion::to<T2>(res_high);

        res_fl2_ptr[offset] = res_fl2;
    }
}

template <typename T>
void launch_gptj_residual_add(T* residual,
                              T* hidden_state,
                              T* attn,
                              T* bias,
                              T* attn_bias,
                              int hidden_dim,
                              int batch,
                              int mp_size,
                              cudaStream_t stream)
{
    int total_count = batch * hidden_dim / 4;
    dim3 block_dims(1024);
    dim3 grid_dims((total_count - 1) / 1024 + 1);  // (batch_size);

    gptj_residual_add<<<grid_dims, block_dims, 0, stream>>>(
        residual, hidden_state, attn, bias, attn_bias, total_count, hidden_dim / 4, 1.0 / mp_size);
}

#define INSTANTIATE_GPT_RES_ADD(T) \
    template void launch_gptj_residual_add<T>(T*, T*, T*, T*, T*, int, int, int, cudaStream_t);

INSTANTIATE_GPT_RES_ADD(float);
INSTANTIATE_GPT_RES_ADD(__half);
#ifdef BF16_AVAILABLE
INSTANTIATE_GPT_RES_ADD(__nv_bfloat16);
#endif

template <typename T>
__global__ void moe_res_matmul(T* residual, T* coef, T* mlp_out, int seq_len, int hidden_dim)
{
    constexpr int granularity = 16;
    constexpr int vals_per_access = granularity / sizeof(T);

    T* residual_seq = residual + blockIdx.x * hidden_dim;
    T* mlp_out_seq = mlp_out + blockIdx.x * hidden_dim;

    for (unsigned tid = threadIdx.x * vals_per_access; tid < hidden_dim;
         tid += blockDim.x * vals_per_access) {
        T mlp[vals_per_access];
        T res[vals_per_access];
        T coef1[vals_per_access];
        T coef2[vals_per_access];

        mem_access::load_global<granularity>(mlp, mlp_out_seq + tid);
        mem_access::load_global<granularity>(res, residual_seq + tid);
        mem_access::load_global<granularity>(coef1, coef + tid);
        mem_access::load_global<granularity>(coef2, coef + tid + hidden_dim);

#pragma unroll
        for (int idx = 0; idx < vals_per_access; idx++) {
            mlp[idx] = mlp[idx] * coef2[idx] + res[idx] * coef1[idx];
        }

        mem_access::store_global<granularity>(mlp_out_seq + tid, mlp);
    }
}

template <typename T>
void launch_moe_res_matmul(T* residual,
                           T* coef,
                           T* mlp_out,
                           int seq_len,
                           int hidden_dim,
                           cudaStream_t stream)
{
    dim3 grid_dim(seq_len);
    dim3 block_dim(1024);
    moe_res_matmul<<<grid_dim, block_dim, 0, stream>>>(
        residual, coef, mlp_out, seq_len, hidden_dim);
}

#define INSTANTIATE_LAUNCH_MOE_RES_MATMUL(T) \
    template void launch_moe_res_matmul<T>(T*, T*, T*, int, int, cudaStream_t);

INSTANTIATE_LAUNCH_MOE_RES_MATMUL(float);
#ifdef BF16_AVAILABLE
INSTANTIATE_LAUNCH_MOE_RES_MATMUL(__nv_bfloat16);
#endif
INSTANTIATE_LAUNCH_MOE_RES_MATMUL(__half);

template <typename T>
__global__ void pad_data_kernel(T* padded_output, T* output, int head_size, int padded_head_size)
{
    using T2 =
        typename std::conditional<std::is_same<T, __half>::value, __half2, __nv_bfloat162>::type;
    float4* padded_output_cast = reinterpret_cast<float4*>(padded_output);
    float4* output_cast = reinterpret_cast<float4*>(output);
    int bid = blockIdx.x * (blockDim.y) + threadIdx.y;
    int idx = threadIdx.x;
    padded_output_cast += (bid * padded_head_size);
    output_cast += (bid * head_size);
    float4 ZERO;
    const T2 zero_h = conversion::to<T2>(0.f);
    T2* ZERO_h = reinterpret_cast<T2*>(&ZERO);
#pragma unroll
    for (int i = 0; i < 4; i++) ZERO_h[i] = zero_h;
    if (idx < head_size)
        padded_output_cast[idx] = output_cast[idx];
    else
        padded_output_cast[idx] = ZERO;
}

__global__ void pad_data_kernel(float* padded_output,
                                float* output,
                                int head_size,
                                int padded_head_size)
{
}

template <typename T>
void pad_data(T* padded_output,
              T* output,
              int bsz,
              int head_size,
              int padded_head_size,
              cudaStream_t stream)
{
    dim3 grid_dim((bsz - 1) / 16 + 1);
    dim3 block_dim(padded_head_size / 8, 16);
    pad_data_kernel<<<grid_dim, block_dim, 0, stream>>>(
        padded_output, output, head_size / 8, padded_head_size / 8);
}

#define INSTANTIATE_PAD_DATA(T) template void pad_data(T*, T*, int, int, int, cudaStream_t stream);

INSTANTIATE_PAD_DATA(float);
INSTANTIATE_PAD_DATA(__half);
#ifdef BF16_AVAILABLE
INSTANTIATE_PAD_DATA(__nv_bfloat16);
#endif

template <typename T>
__global__ void pad_head_seq_kernel(T* padded_output,
                                    T* output,
                                    int seq_len,
                                    int padded_seq_len,
                                    int head_size,
                                    int padded_head_size)
{
    using T2 =
        typename std::conditional<std::is_same<T, __half>::value, __half2, __nv_bfloat162>::type;
    float4* padded_output_cast = reinterpret_cast<float4*>(padded_output);
    float4* output_cast = reinterpret_cast<float4*>(output);
    int bsz = blockIdx.x;
    int bid = blockIdx.y * (blockDim.y) + threadIdx.y;
    int idx = threadIdx.x;
    padded_output_cast += (bsz * padded_seq_len + bid) * padded_head_size;
    output_cast += (bsz * seq_len + bid) * head_size;
    float4 ZERO;
    const T2 zero_h = conversion::to<T2>(0.f);
    T2* ZERO_h = reinterpret_cast<T2*>(&ZERO);
#pragma unroll
    for (int i = 0; i < 4; i++) ZERO_h[i] = zero_h;

    if (idx < head_size && bid < seq_len)
        padded_output_cast[idx] = output_cast[idx];
    else
        padded_output_cast[idx] = ZERO;
}

__global__ void pad_head_seq_kernel(float* padded_output,
                                    float* output,
                                    int seq_len,
                                    int padded_seq_len,
                                    int head_size,
                                    int padded_head_size)
{
}

template <typename T>
void pad_head_seq(T* padded_output,
                  T* output,
                  int bsz,
                  int seq_len,
                  int padded_seq_len,
                  int head_size,
                  int padded_head_size,
                  cudaStream_t stream)
{
    dim3 grid_dim(bsz, padded_seq_len / 16);
    dim3 block_dim(padded_head_size / 8, 16);
    pad_head_seq_kernel<<<grid_dim, block_dim, 0, stream>>>(
        padded_output, output, seq_len, padded_seq_len, head_size / 8, padded_head_size / 8);
}

#define INSTANTIATE_PAD_HEAD_SEQ(T) \
    template void pad_head_seq<T>(T*, T*, int, int, int, int, int, cudaStream_t);

INSTANTIATE_PAD_HEAD_SEQ(__half);
#ifdef BF16_AVAILABLE
INSTANTIATE_PAD_HEAD_SEQ(__nv_bfloat16);
#endif
INSTANTIATE_PAD_HEAD_SEQ(float);

// TODO(cmikeh2): evaluate different GeLU performance
__device__ __forceinline__ float old_gelu(float val)
{
    // 1 / sqrt(2)
    constexpr float rsqrt_2 = 0.707106769084930419922;
    return val * 0.5f * (1.0f + erff(val * rsqrt_2));
}

namespace fused_geglu {
constexpr int threads = 256;
constexpr int steps = 2;
constexpr int granularity = 16;
}  // namespace fused_geglu

__device__ __forceinline__ float silu(float val) { return val / (1.0f + expf(-val)); }

template <typename T, bool useGelu>
__global__ void fused_gate_activation(T* output,
                                      const T* activation,
                                      const T* bias,
                                      int base_channels,
                                      int output_stride,
                                      int total_elems)
{
    constexpr int T_per_access = fused_geglu::granularity / sizeof(T);
    constexpr int T_per_step = T_per_access * fused_geglu::threads;
    constexpr int T_per_block = T_per_step * fused_geglu::steps;

    const int id = blockIdx.x * T_per_block + threadIdx.x * T_per_access;

#pragma unroll
    for (int i = 0; i < fused_geglu::steps; i++) {
        T activation_buffer_1[T_per_access];
        T activation_buffer_2[T_per_access];
        T bias_buffer_1[T_per_access];
        T bias_buffer_2[T_per_access];

        const int iter_id = id + T_per_step * i;
        if (iter_id < total_elems) {
            const int channel_id = iter_id % base_channels;
            const int seq_id = iter_id / base_channels;
            const int seq_offset = seq_id * base_channels * 2;

            mem_access::load_global<fused_geglu::granularity>(activation_buffer_1,
                                                              activation + seq_offset + channel_id);
            mem_access::load_global<fused_geglu::granularity>(
                activation_buffer_2, activation + seq_offset + channel_id + base_channels);
            mem_access::load_global<fused_geglu::granularity>(
                bias_buffer_1, bias + channel_id, bias != nullptr);
            mem_access::load_global<fused_geglu::granularity>(
                bias_buffer_2, bias + channel_id + base_channels, bias != nullptr);

            // Since the GeLU is going to happen at float, might as well
            // convert
#pragma unroll
            for (int v = 0; v < T_per_access; v++) {
                T hidden_state = activation_buffer_1[v] + bias_buffer_1[v];
                T pre_gate = activation_buffer_2[v] + bias_buffer_2[v];
                float pre_gate_f = conversion::to<float>(pre_gate);
                float gate_f = (useGelu) ? old_gelu(pre_gate_f) : silu(pre_gate_f);
                T gate = conversion::to<T>(gate_f);
                activation_buffer_1[v] = hidden_state * gate;
            }

            mem_access::store_global<fused_geglu::granularity>(
                output + seq_id * output_stride + channel_id, activation_buffer_1);
        }
    }
}

template <typename T>
void launch_gated_activation(T* output,
                             const T* activation,
                             const T* bias,
                             int rows,
                             int output_stride,
                             int elems_per_row,
                             bool use_gelu,
                             cudaStream_t stream)
{
    /*
    Fused bias GEGLU is a variant of the gated activation functions.
    The input here is a matrix of [batch, seq_len, 2 * intermediate_dim]
    where the second half of the channels act as GeLU gates for the first
    half.
    */

    // Re-derive the above figures
    constexpr int T_per_access = fused_geglu::granularity / sizeof(T);
    constexpr int T_per_step = T_per_access * fused_geglu::threads;
    constexpr int T_per_block = T_per_step * fused_geglu::steps;

    const int base_channels = elems_per_row / 2;
    const int total_elems = base_channels * rows;

    dim3 block(fused_geglu::threads);
    dim3 grid((total_elems + T_per_block - 1) / T_per_block);

    if (use_gelu) {
        fused_gate_activation<T, true><<<grid, block, 0, stream>>>(
            output, activation, bias, base_channels, output_stride, total_elems);
    } else {
        fused_gate_activation<T, false><<<grid, block, 0, stream>>>(
            output, activation, bias, base_channels, output_stride, total_elems);
    }
}

#define INSTANTIATE_LAUNCH_GATED_ACTIVATION(T) \
    template void launch_gated_activation(     \
        T*, const T*, const T*, int, int, int, bool, cudaStream_t);

INSTANTIATE_LAUNCH_GATED_ACTIVATION(__half);
#ifdef BF16_AVAILABLE
INSTANTIATE_LAUNCH_GATED_ACTIVATION(__nv_bfloat16);
#endif
INSTANTIATE_LAUNCH_GATED_ACTIVATION(float);
