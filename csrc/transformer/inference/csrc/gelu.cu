#include "custom_cuda_layers.h"

inline __device__ float gelu(const float x)
{
    const float sqrt_param = 0.79788456080286535587989211986876f;
    const float mul_param = 0.044715;
    return x * 0.5f * (1.0f + tanhf(sqrt_param * (x + mul_param * x * x * x)));
}

__global__ void fused_bias_gelu(float* input,
                                const float* bias,
                                int total_count,
                                int intermediate_size)
{
    float4* input_cast = reinterpret_cast<float4*>(input);
    const float4* bias_cast = reinterpret_cast<const float4*>(bias);
    int offset = blockIdx.x * blockDim.x + threadIdx.x;

    if (offset < total_count) {
        float4 data = input_cast[offset];
        float4 bias_data = bias_cast[offset % intermediate_size];

        data.x += bias_data.x;
        data.y += bias_data.y;
        data.z += bias_data.z;
        data.w += bias_data.w;

        data.x = gelu(data.x);
        data.y = gelu(data.y);
        data.z = gelu(data.z);
        data.w = gelu(data.w);

        input_cast[offset] = data;
    }
}

__global__ void fused_bias_gelu(__half* input,
                                const __half* bias,
                                int total_count,
                                int intermediate_size)
{
#if __CUDA_ARCH__ >= 700

    float2* input_cast = reinterpret_cast<float2*>(input);
    const float2* bias_cast = reinterpret_cast<const float2*>(bias);

    int offset = blockIdx.x * blockDim.x + threadIdx.x;

    if (offset < total_count) {
        float2 vals_vec = input_cast[offset];
        float2 bias_vec = bias_cast[offset % intermediate_size];

        __half2* vals_half = reinterpret_cast<__half2*>(&vals_vec);
        __half2* bias_half = reinterpret_cast<__half2*>(&bias_vec);

        float2 low_data = __half22float2(vals_half[0]);
        float2 high_data = __half22float2(vals_half[1]);

        float2 low_bias = __half22float2(bias_half[0]);
        float2 high_bias = __half22float2(bias_half[1]);

        low_data.x += low_bias.x;
        low_data.y += low_bias.y;
        high_data.x += high_bias.x;
        high_data.y += high_bias.y;

        low_data.x = gelu(low_data.x);
        low_data.y = gelu(low_data.y);
        high_data.x = gelu(high_data.x);
        high_data.y = gelu(high_data.y);

        vals_half[0] = __float22half2_rn(low_data);
        vals_half[1] = __float22half2_rn(high_data);

        input_cast[offset] = vals_vec;
    }
#endif
}

template <typename T>
void launch_bias_gelu(T* input,
                      const T* bias,
                      int intermediate_size,
                      int batch_size,
                      cudaStream_t stream)
{
    int total_count = batch_size * (intermediate_size / 4);
    int threads = 1024;  // intermediate_size / iterations / 4;
    dim3 block_dims(threads);
    dim3 grid_dims(((total_count - 1) / 1024 + 1));  // (batch_size);

    fused_bias_gelu<<<grid_dims, block_dims, 0, stream>>>(
        input, bias, total_count, intermediate_size / 4);
}

template void launch_bias_gelu<float>(float*, const float*, int, int, cudaStream_t);
template void launch_bias_gelu<__half>(__half*, const __half*, int, int, cudaStream_t);

__global__ void fused_bias_add(float* input, const float* bias, int total_count, int hidden_size)
{
    float4* input_cast = reinterpret_cast<float4*>(input);
    const float4* bias_cast = reinterpret_cast<const float4*>(bias);
    int offset = blockIdx.x * blockDim.x + threadIdx.x;

    if (offset < total_count) {
        float4 data = input_cast[offset];
        float4 bias_data = bias_cast[offset % hidden_size];

        data.x += bias_data.x;
        data.y += bias_data.y;
        data.z += bias_data.z;
        data.w += bias_data.w;

        input_cast[offset] = data;
    }
}

__global__ void fused_bias_add(__half* input, const __half* bias, int total_count, int hidden_size)
{
#if __CUDA_ARCH__ >= 700

    float2* input_cast = reinterpret_cast<float2*>(input);
    const float2* bias_cast = reinterpret_cast<const float2*>(bias);

    int offset = blockIdx.x * blockDim.x + threadIdx.x;

    if (offset < total_count) {
        float2 vals_vec = input_cast[offset];
        float2 bias_vec = bias_cast[offset % hidden_size];

        __half2* vals_half = reinterpret_cast<__half2*>(&vals_vec);
        __half2* bias_half = reinterpret_cast<__half2*>(&bias_vec);

        float2 low_data = __half22float2(vals_half[0]);
        float2 high_data = __half22float2(vals_half[1]);

        float2 low_bias = __half22float2(bias_half[0]);
        float2 high_bias = __half22float2(bias_half[1]);

        low_data.x += low_bias.x;
        low_data.y += low_bias.y;
        high_data.x += high_bias.x;
        high_data.y += high_bias.y;

        vals_half[0] = __float22half2_rn(low_data);
        vals_half[1] = __float22half2_rn(high_data);

        input_cast[offset] = vals_vec;
    }
#endif
}

template <typename T>
void launch_bias_add(T* input, const T* bias, int hidden_size, int batch_size, cudaStream_t stream)
{
    int total_count = batch_size * (hidden_size / 4);
    int threads = 1024;  // hidden_size / iterations / 4;
    dim3 block_dims(threads);
    dim3 grid_dims(((total_count - 1) / threads + 1));  // (batch_size);

    fused_bias_add<<<grid_dims, block_dims, 0, stream>>>(input, bias, total_count, hidden_size / 4);
}

template void launch_bias_add<float>(float*, const float*, int, int, cudaStream_t);
template void launch_bias_add<__half>(__half*, const __half*, int, int, cudaStream_t);

__global__ void fused_bias_residual(float* input,
                                    const float* residual,
                                    const float* bias,
                                    int total_count,
                                    int intermediate_size)
{
    float4* input_cast = reinterpret_cast<float4*>(input);
    const float4* residual_cast = reinterpret_cast<const float4*>(residual);
    const float4* bias_cast = reinterpret_cast<const float4*>(bias);
    int offset = blockIdx.x * blockDim.x + threadIdx.x;

    if (offset < total_count) {
        float4 data = input_cast[offset];
        float4 res_vec = residual_cast[offset];
        float4 bias_data = bias_cast[offset % intermediate_size];

        data.x += (res_vec.x + bias_data.x);
        data.y += (res_vec.y + bias_data.y);
        data.z += (res_vec.z + bias_data.z);
        data.w += (res_vec.w + bias_data.w);

        input_cast[offset] = data;
    }
}

__global__ void fused_bias_residual(__half* input,
                                    const __half* residual,
                                    const __half* bias,
                                    int total_count,
                                    int intermediate_size)
{
#if __CUDA_ARCH__ >= 700

    float2* input_cast = reinterpret_cast<float2*>(input);
    const float2* residual_cast = reinterpret_cast<const float2*>(residual);

    const float2* bias_cast = reinterpret_cast<const float2*>(bias);

    int offset = blockIdx.x * blockDim.x + threadIdx.x;

    if (offset < total_count) {
        float2 vals_vec = input_cast[offset];
        float2 res_vec = residual_cast[offset];

        float2 bias_vec = bias_cast[offset % intermediate_size];

        __half2* vals_half = reinterpret_cast<__half2*>(&vals_vec);
        __half2* res_half = reinterpret_cast<__half2*>(&res_vec);
        __half2* bias_half = reinterpret_cast<__half2*>(&bias_vec);

        float2 low_data = __half22float2(vals_half[0]);
        float2 high_data = __half22float2(vals_half[1]);

        float2 low_res = __half22float2(res_half[0]);
        float2 high_res = __half22float2(res_half[1]);

        float2 low_bias = __half22float2(bias_half[0]);
        float2 high_bias = __half22float2(bias_half[1]);

        low_data.x += (low_res.x + low_bias.x);
        low_data.y += (low_res.y + low_bias.y);
        high_data.x += (high_res.x + high_bias.x);
        high_data.y += (high_res.y + high_bias.y);

        vals_half[0] = __float22half2_rn(low_data);
        vals_half[1] = __float22half2_rn(high_data);

        input_cast[offset] = vals_vec;
    }
#endif
}

template <typename T>
void launch_bias_residual(T* input,
                          const T* residual,
                          const T* bias,
                          int batch,
                          int intermediate_size,
                          cudaStream_t stream)
{
    int total_count = batch * intermediate_size / 4;
    dim3 block_dims(1024);
    dim3 grid_dims((total_count - 1) / 1024 + 1);  // (batch_size);

    fused_bias_residual<<<grid_dims, block_dims, 0, stream>>>(
        input, residual, bias, total_count, intermediate_size / 4);
}

template void launch_bias_residual<float>(float*,
                                          const float*,
                                          const float*,
                                          int,
                                          int,
                                          cudaStream_t);
template void launch_bias_residual<__half>(__half*,
                                           const __half*,
                                           const __half*,
                                           int,
                                           int,
                                           cudaStream_t);

__global__ void gptj_residual_add(float* input,
                                  float* output,
                                  float* attn,
                                  float* bias,
                                  int total_count,
                                  int intermediate_size)
{
    float4* input_cast = reinterpret_cast<float4*>(input);
    float4* output_cast = reinterpret_cast<float4*>(output);
    float4* attn_cast = reinterpret_cast<float4*>(attn);
    float4* bias_cast = reinterpret_cast<float4*>(bias);
    int offset = blockIdx.x * blockDim.x + threadIdx.x;

    if (offset < total_count) {
        float4 data = input_cast[offset];
        float4 out = output_cast[offset];
        float4 res_vec = attn_cast[offset];
        float4 bias_data = bias_cast[offset % intermediate_size];

        data.x += (out.x + res_vec.x + bias_data.x);
        data.y += (out.y + res_vec.y + bias_data.y);
        data.z += (out.z + res_vec.z + bias_data.z);
        data.w += (out.w + res_vec.w + bias_data.w);

        output_cast[offset] = data;
    }
}

__global__ void gptj_residual_add(__half* input,
                                  __half* output,
                                  __half* attn,
                                  __half* bias,
                                  int total_count,
                                  int intermediate_size)
{
#if __CUDA_ARCH__ >= 700

    float2* input_cast = reinterpret_cast<float2*>(input);
    float2* output_cast = reinterpret_cast<float2*>(output);
    float2* attn_cast = reinterpret_cast<float2*>(attn);

    float2* bias_cast = reinterpret_cast<float2*>(bias);

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

        low_data.x += (low_out.x + low_res.x + low_bias.x);
        low_data.y += (low_out.y + low_res.y + low_bias.y);
        high_data.x += (high_out.x + high_res.x + high_bias.x);
        high_data.y += (high_out.y + high_res.y + high_bias.y);

        vals_half[0] = __float22half2_rn(low_data);
        vals_half[1] = __float22half2_rn(high_data);

        output_cast[offset] = vals_vec;
    }
#endif
}

template <typename T>
void launch_gptj_residual_add(T* input,
                              T* output,
                              T* attn,
                              T* bias,
                              int hidden_dim,
                              int batch,
                              cudaStream_t stream)
{
    int total_count = batch * hidden_dim / 4;
    dim3 block_dims(1024);
    dim3 grid_dims((total_count - 1) / 1024 + 1);  // (batch_size);

    gptj_residual_add<<<grid_dims, block_dims, 0, stream>>>(
        input, output, attn, bias, total_count, hidden_dim / 4);
}

template void
launch_gptj_residual_add<float>(float*, float*, float*, float*, int, int, cudaStream_t);
template void
launch_gptj_residual_add<__half>(__half*, __half*, __half*, __half*, int, int, cudaStream_t);
