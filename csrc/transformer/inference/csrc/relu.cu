/*
Copyright 2022 The Microsoft DeepSpeed Team
*/

#include "inference_cuda_layers.h"
#include "memory_access_utils.h"

namespace cg = cooperative_groups;
#define MAX_CAP 4
#define MAX_SEQ 2048

inline __device__ float relu(const float x) { return x < 0 ? 0 : x; }

__global__ void fused_bias_relu(float* input,
                                const float* bias,
                                int total_count,
                                int intermediate_size)
{
    // Input restriction: intermediate_size % vals_per_access == 0
    constexpr int granularity = 16;
    constexpr int vals_per_access = granularity / sizeof(float);
    const int offset = (blockIdx.x * blockDim.x + threadIdx.x) * vals_per_access;

    if (offset < total_count) {
        float data[vals_per_access];
        float data_bias[vals_per_access];
        mem_access::load_global<granularity>(data, input + offset);
        mem_access::load_global<granularity>(data_bias, bias + (offset % intermediate_size));

#pragma unroll
        for (int i = 0; i < vals_per_access; i++) { data[i] = relu(data[i] + data_bias[i]); }

        mem_access::store_global<granularity>(input + offset, data);
    }
}

__global__ void fused_bias_relu(__half* input,
                                const __half* bias,
                                int total_count,
                                int intermediate_size)
{
    // Input restriction: intermediate_size % vals_per_access == 0
    // This kernel doubles the per-thread ALU workload as compared to the float implementation
#ifdef HALF_PRECISION_AVAILABLE
    constexpr int granularity = 16;
    constexpr int vals_per_access = granularity / sizeof(__half);
    int offset = (blockIdx.x * blockDim.x + threadIdx.x) * vals_per_access;

    if (offset < total_count) {
        // Divide by 2 since we store two values per __half2
        __half2 data[vals_per_access / 2];
        __half2 bias_data[vals_per_access / 2];
        mem_access::load_global<granularity>(data, input + offset);
        mem_access::load_global<granularity>(bias_data, bias + (offset % intermediate_size));

#pragma unroll
        for (int i = 0; i < vals_per_access / 2; i++) {
            float2 data_f = __half22float2(data[i]);
            float2 bias_f = __half22float2(bias_data[i]);
            data[i] = __floats2half2_rn(relu(data_f.x + bias_f.x), relu(data_f.y + bias_f.y));
        }

        mem_access::store_global<granularity>(input + offset, data);
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
    constexpr int threads = 1024;
    constexpr int granularity = 16;

    const int total_count = batch_size * intermediate_size;
    const int elems_per_block = threads * (granularity / sizeof(T));
    dim3 block_dims(threads);
    dim3 grid_dims((total_count + elems_per_block - 1) / elems_per_block);

    fused_bias_relu<<<grid_dims, block_dims, 0, stream>>>(
        input, bias, total_count, intermediate_size);
}

template void launch_bias_relu<float>(float*, const float*, int, int, cudaStream_t);
template void launch_bias_relu<__half>(__half*, const __half*, int, int, cudaStream_t);
