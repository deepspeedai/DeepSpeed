#include "custom_cuda_layers.h"

#define MAX_CAP 4
#define MAX_SEQ 2048

inline __device__ float relu(const float x) { return x < 0 ? 0 : x; }

__global__ void fused_bias_relu(float* input,
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

        data.x = relu(data.x);
        data.y = relu(data.y);
        data.z = relu(data.z);
        data.w = relu(data.w);

        input_cast[offset] = data;
    }
}

__global__ void fused_bias_relu(__half* input,
                                const __half* bias,
                                int total_count,
                                int intermediate_size)
{
#ifdef HALF_PRECISION_AVAILABLE

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

        low_data.x = relu(low_data.x);
        low_data.y = relu(low_data.y);
        high_data.x = relu(high_data.x);
        high_data.y = relu(high_data.y);

        vals_half[0] = __float22half2_rn(low_data);
        vals_half[1] = __float22half2_rn(high_data);

        input_cast[offset] = vals_vec;
    }
#endif
}

template <typename T>
void launch_bias_relu(T* input,
                      const T* bias,
                      int intermediate_size,
                      int batch_size,
                      cudaStream_t stream)
{
    int total_count = batch_size * (intermediate_size / 4);
    int threads = 1024;  // intermediate_size / iterations / 4;
    dim3 block_dims(threads);
    dim3 grid_dims(((total_count - 1) / 1024 + 1));  // (batch_size);

    fused_bias_relu<<<grid_dims, block_dims, 0, stream>>>(
        input, bias, total_count, intermediate_size / 4);
}

template void launch_bias_relu<float>(float*, const float*, int, int, cudaStream_t);
template void launch_bias_relu<__half>(__half*, const __half*, int, int, cudaStream_t);
