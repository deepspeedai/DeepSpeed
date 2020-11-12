

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
