/*
Copyright 2022 The Microsoft DeepSpeed Team
*/

#include "conversion_utils.h"
#include "inference_cuda_layers.h"
#include "memory_access_utils.h"

namespace cg = cooperative_groups;
#define MAX_CAP 4
#define MAX_SEQ 2048

inline __device__ float gelu(const float x)
{
    const float sqrt_param = 0.79788456080286535587989211986876f;
    const float mul_param = 0.044715;
    return x * 0.5f * (1.0f + tanhf(sqrt_param * (x + mul_param * x * x * x)));
}

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
        mem_access::load_global<granularity>(data_bias, bias + (offset % intermediate_size));

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

template void launch_bias_gelu<float>(float*, const float*, int, int, cudaStream_t);
template void launch_bias_gelu<__half>(__half*, const __half*, int, int, cudaStream_t);

// Not called directly from DeepSpeed, but used in ds_qkv_gemm_int8, ds_linear_layer, etc.
__global__ void fused_bias_add(float* input, const float* bias, int total_count, int hidden_size)
{
    constexpr int granularity = 16;
    constexpr int vals_per_access = granularity / sizeof(float);
    const int offset = (blockIdx.x * blockDim.x + threadIdx.x) * vals_per_access;

    if (offset < total_count) {
        float data[vals_per_access];
        float bias_data[vals_per_access];
        mem_access::load_global<granularity>(data, input + offset);
        mem_access::load_global<granularity>(bias_data, bias + (offset % hidden_size));

#pragma unroll
        for (int i = 0; i < vals_per_access; i++) { data[i] += bias_data[i]; }

        mem_access::store_global<granularity>(input + offset, data);
    }
}

__global__ void fused_bias_add(__half* input, const __half* bias, int total_count, int hidden_size)
{
#ifdef HALF_PRECISION_AVAILABLE
    constexpr int granularity = 16;
    constexpr int vals_per_access = granularity / sizeof(__half);
    const int offset = (blockIdx.x * blockDim.x + threadIdx.x) * vals_per_access;

    if (offset < total_count) {
        __half2 data[vals_per_access / 2];
        __half2 bias_data[vals_per_access / 2];
        mem_access::load_global<granularity>(data, input + offset);
        mem_access::load_global<granularity>(bias_data, bias + (offset % hidden_size));

#pragma unroll
        for (int i = 0; i < vals_per_access / 2; i++) {
            float2 data_f = __half22float2(data[i]);
            float2 bias_f = __half22float2(bias_data[i]);
            data[i] = __floats2half2_rn(data_f.x + bias_f.x, data_f.y + bias_f.y);
        }

        mem_access::store_global<granularity>(input + offset, data);
    }
#endif
}

template <typename T>
void launch_bias_add(T* input, const T* bias, int hidden_size, int batch_size, cudaStream_t stream)
{
    constexpr int threads = 1024;
    constexpr int granularity = 16;

    const int total_count = batch_size * hidden_size;
    const int elems_per_block = threads * (granularity / sizeof(T));
    dim3 block_dims(threads);
    dim3 grid_dims((total_count + elems_per_block - 1) / elems_per_block);

    fused_bias_add<<<grid_dims, block_dims, 0, stream>>>(input, bias, total_count, hidden_size);
}

template void launch_bias_add<float>(float*, const float*, int, int, cudaStream_t);
template void launch_bias_add<__half>(__half*, const __half*, int, int, cudaStream_t);

__global__ void fused_bias_residual(float* input,
                                    float* output,
                                    float* attn,
                                    float* bias,
                                    float* attnbias,
                                    int total_count,
                                    int intermediate_size,
                                    float mp_scale,
                                    bool preln)
{
    float4* input_cast = reinterpret_cast<float4*>(input);
    float4* output_cast = reinterpret_cast<float4*>(output);
    float4* attn_cast = reinterpret_cast<float4*>(attn);
    float4* bias_cast = reinterpret_cast<float4*>(bias);
    float4* attnbias_cast = reinterpret_cast<float4*>(attnbias);
    int offset = blockIdx.x * blockDim.x + threadIdx.x;

    if (offset < total_count) {
        float4 data = input_cast[offset];
        float4 out = output_cast[offset];
        float4 res_vec = attn_cast[offset];
        float4 bias_data = bias_cast[offset % intermediate_size];
        float4 attn_bias = attnbias_cast[offset % intermediate_size];
        if (preln) {
            data.x = (data.x + res_vec.x + bias_data.x + attn_bias.x) * mp_scale + (out.x);
            data.y = (data.y + res_vec.y + bias_data.y + attn_bias.y) * mp_scale + (out.y);
            data.z = (data.z + res_vec.z + bias_data.z + attn_bias.z) * mp_scale + (out.z);
            data.w = (data.w + res_vec.w + bias_data.w + attn_bias.w) * mp_scale + (out.w);
        } else {
            data.x = data.x + out.x + bias_data.x;
            data.y = data.y + out.y + bias_data.y;
            data.z = data.z + out.z + bias_data.z;
            data.w = data.w + out.w + bias_data.w;
        }
        input_cast[offset] = data;
    }
}

__global__ void fused_bias_residual(__half* input,
                                    __half* output,
                                    __half* attn,
                                    __half* bias,
                                    __half* attn_bias,
                                    int total_count,
                                    int intermediate_size,
                                    float mp_scale,
                                    bool preln)
{
#ifdef HALF_PRECISION_AVAILABLE

    float2* input_cast = reinterpret_cast<float2*>(input);
    float2* output_cast = reinterpret_cast<float2*>(output);
    float2* attn_cast = reinterpret_cast<float2*>(attn);

    float2* bias_cast = reinterpret_cast<float2*>(bias);
    float2* attnbias_cast = reinterpret_cast<float2*>(attn_bias);

    int offset = blockIdx.x * blockDim.x + threadIdx.x;

    if (offset < total_count) {
        float2 vals_vec = input_cast[offset];
        float2 out_vec = output_cast[offset];
        float2 res_vec = attn_cast[offset];

        float2 bias_vec = bias_cast[offset % intermediate_size];
        float2 attn_bias_vec = attnbias_cast[offset % intermediate_size];

        __half2* vals_half = reinterpret_cast<__half2*>(&vals_vec);
        __half2* out_half = reinterpret_cast<__half2*>(&out_vec);
        __half2* res_half = reinterpret_cast<__half2*>(&res_vec);
        __half2* bias_half = reinterpret_cast<__half2*>(&bias_vec);
        __half2* attnbias_half = reinterpret_cast<__half2*>(&attn_bias_vec);

        float2 low_data = __half22float2(vals_half[0]);
        float2 high_data = __half22float2(vals_half[1]);

        float2 low_out = __half22float2(out_half[0]);
        float2 high_out = __half22float2(out_half[1]);

        float2 low_res = __half22float2(res_half[0]);
        float2 high_res = __half22float2(res_half[1]);

        float2 low_bias = __half22float2(bias_half[0]);
        float2 high_bias = __half22float2(bias_half[1]);

        float2 attn_low_bias = __half22float2(attnbias_half[0]);
        float2 attn_high_bias = __half22float2(attnbias_half[1]);

        if (preln) {
            low_data.x =
                (low_data.x + low_res.x + (low_bias.x + attn_low_bias.x)) * mp_scale + low_out.x;
            low_data.y =
                (low_data.y + low_res.y + (low_bias.y + attn_low_bias.y)) * mp_scale + low_out.y;
            high_data.x = (high_data.x + high_res.x + (high_bias.x + attn_high_bias.x)) * mp_scale +
                          high_out.x;
            high_data.y = (high_data.y + high_res.y + (high_bias.y + attn_high_bias.y)) * mp_scale +
                          high_out.y;
        } else {
            low_data.x = (low_data.x + low_out.x + low_bias.x);
            low_data.y = (low_data.y + low_out.y + low_bias.y);
            high_data.x = (high_data.x + high_out.x + high_bias.x);
            high_data.y = (high_data.y + high_out.y + high_bias.y);
        }
        vals_half[0] = __float22half2_rn(low_data);
        vals_half[1] = __float22half2_rn(high_data);

        input_cast[offset] = vals_vec;
    }
#endif
}

template <typename T>
void launch_bias_residual(T* input,
                          T* output,
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

    fused_bias_residual<<<grid_dims, block_dims, 0, stream>>>(
        input, output, attn, bias, attn_bias, total_count, hidden_dim / 4, 1.0 / mp_size, preln);
}

template void launch_bias_residual<
    float>(float*, float*, float*, float*, float*, int, int, int, bool, cudaStream_t);
template void launch_bias_residual<
    __half>(__half*, __half*, __half*, __half*, __half*, int, int, int, bool, cudaStream_t);

__global__ void gptj_residual_add(float* input,
                                  float* output,
                                  float* attn,
                                  float* bias,
                                  float* attnbias,
                                  int total_count,
                                  int intermediate_size,
                                  float mp_scale)
{
    float4* input_cast = reinterpret_cast<float4*>(input);
    float4* output_cast = reinterpret_cast<float4*>(output);
    float4* attn_cast = reinterpret_cast<float4*>(attn);
    float4* bias_cast = reinterpret_cast<float4*>(bias);
    float4* attnbias_cast = reinterpret_cast<float4*>(attnbias);
    int offset = blockIdx.x * blockDim.x + threadIdx.x;

    if (offset < total_count) {
        float4 data = input_cast[offset];
        float4 out = output_cast[offset];
        float4 res_vec = attn_cast[offset];
        float4 bias_data = bias_cast[offset % intermediate_size];

        if (attnbias) {
            float4 attn_bias = attnbias_cast[offset % intermediate_size];
            data.x += attn_bias.x;
            data.y += attn_bias.y;
            data.z += attn_bias.z;
            data.w += attn_bias.w;
        }
        data.x = out.x + res_vec.x + (data.x + bias_data.x) * mp_scale;
        data.y = out.y + res_vec.y + (data.y + bias_data.y) * mp_scale;
        data.z = out.z + res_vec.z + (data.z + bias_data.z) * mp_scale;
        data.w = out.w + res_vec.w + (data.w + bias_data.w) * mp_scale;

        input_cast[offset] = data;
    }
}

__global__ void gptj_residual_add(__half* input,
                                  __half* output,
                                  __half* attn,
                                  __half* bias,
                                  __half* attn_bias,
                                  int total_count,
                                  int intermediate_size,
                                  float mp_scale)
{
#ifdef HALF_PRECISION_AVAILABLE

    float2* input_cast = reinterpret_cast<float2*>(input);
    float2* output_cast = reinterpret_cast<float2*>(output);
    float2* attn_cast = reinterpret_cast<float2*>(attn);

    float2* bias_cast = reinterpret_cast<float2*>(bias);
    float2* attnbias_cast = reinterpret_cast<float2*>(attn_bias);

    int offset = blockIdx.x * blockDim.x + threadIdx.x;

    if (offset < total_count) {
        float2 vals_vec = input_cast[offset];
        float2 out_vec = output_cast[offset];
        float2 res_vec = attn_cast[offset];

        float2 bias_vec = bias_cast[offset % intermediate_size];

        __half2* vals_half = reinterpret_cast<__half2*>(&vals_vec);
        __half2* out_half = reinterpret_cast<__half2*>(&out_vec);
        __half2* res_half = reinterpret_cast<__half2*>(&res_vec);
        __half2* bias_half = reinterpret_cast<__half2*>(&bias_vec);

        float2 low_data = __half22float2(vals_half[0]);
        float2 high_data = __half22float2(vals_half[1]);

        float2 low_out = __half22float2(out_half[0]);
        float2 high_out = __half22float2(out_half[1]);

        float2 low_res = __half22float2(res_half[0]);
        float2 high_res = __half22float2(res_half[1]);

        float2 low_bias = __half22float2(bias_half[0]);
        float2 high_bias = __half22float2(bias_half[1]);
        if (attn_bias) {
            float2 attn_bias_vec = attnbias_cast[offset % intermediate_size];
            __half2* attnbias_half = reinterpret_cast<__half2*>(&attn_bias_vec);
            float2 attn_low_bias = __half22float2(attnbias_half[0]);
            float2 attn_high_bias = __half22float2(attnbias_half[1]);
            low_data.x += attn_low_bias.x;
            low_data.y += attn_low_bias.y;
            high_data.x += attn_high_bias.x;
            high_data.y += attn_high_bias.y;
        }

        low_data.x = low_res.x + low_out.x + (low_data.x + low_bias.x) * mp_scale;
        low_data.y = low_res.y + low_out.y + (low_data.y + low_bias.y) * mp_scale;
        high_data.x = high_res.x + high_out.x + (high_data.x + high_bias.x) * mp_scale;
        high_data.y = high_res.y + high_out.y + (high_data.y + high_bias.y) * mp_scale;

        vals_half[0] = __float22half2_rn(low_data);
        vals_half[1] = __float22half2_rn(high_data);

        input_cast[offset] = vals_vec;
    }
#endif
}

template <typename T>
void launch_gptj_residual_add(T* input,
                              T* output,
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
        input, output, attn, bias, attn_bias, total_count, hidden_dim / 4, 1.0 / mp_size);
}

template void launch_gptj_residual_add<float>(float*,
                                              float*,
                                              float*,
                                              float*,
                                              float*,
                                              int,
                                              int,
                                              int,
                                              cudaStream_t);
template void launch_gptj_residual_add<__half>(__half*,
                                               __half*,
                                               __half*,
                                               __half*,
                                               __half*,
                                               int,
                                               int,
                                               int,
                                               cudaStream_t);
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

template void launch_moe_res_matmul(float* residual,
                                    float* coef,
                                    float* mlp_out,
                                    int seq_len,
                                    int hidden_dim,
                                    cudaStream_t stream);
template void launch_moe_res_matmul(__half* residual,
                                    __half* coef,
                                    __half* mlp_out,
                                    int seq_len,
                                    int hidden_dim,
                                    cudaStream_t stream);
