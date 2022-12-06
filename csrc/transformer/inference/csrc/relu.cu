/*
Copyright 2022 The Microsoft DeepSpeed Team
*/

#include "conversion_utils.h"
#include "inference_cuda_layers.h"
#include "memory_access_utils.h"

namespace cg = cooperative_groups;
#define MAX_CAP 4
#define MAX_SEQ 2048

inline __device__ float relu(const float x) { return x < 0 ? 0 : x; }

/*
In-place relu(biasAdd(x)) for channels last
*/
template <typename T>
__global__ void fused_bias_relu(T* input, const T* bias, int total_count, int intermediate_size)
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
            data[i] = conversion::to<T>(relu(data_f + bias_f));
        }

        mem_access::store_global<granularity>(input + offset, data);
    }
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
