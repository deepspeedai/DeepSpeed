

#include "custom_cuda_layers.h"

__global__ void param_update_kernel(const float* input, __half* output, int size)
{
    const float4* input_cast = reinterpret_cast<const float4*>(input);
    float2* output_cast = reinterpret_cast<float2*>(output);

    int id = blockIdx.x * blockDim.x + threadIdx.x;

    if (id < size) {
        float4 data = input_cast[id];
        float2 cast_data;
        __half* output_h = reinterpret_cast<__half*>(&cast_data);

        output_h[0] = (__half)data.x;
        output_h[1] = (__half)data.y;
        output_h[2] = (__half)data.z;
        output_h[3] = (__half)data.w;

        output_cast[id] = cast_data;
    }
}

void launch_param_update(const float* input, __half* output, int size, cudaStream_t stream)
{
    int threads = 512;

    size /= 4;
    dim3 grid_dim((size - 1) / threads + 1);
    dim3 block_dim(threads);

    param_update_kernel<<<grid_dim, block_dim, 0, stream>>>(input, output, size);
}
