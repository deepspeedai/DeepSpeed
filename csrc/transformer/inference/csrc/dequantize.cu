#include "custom_cuda_layers.h"

#define MAX_QUANTIZE_GROUPING 1024

#define loop_unroll 1
#define loop_unroll_bits 1

__global__ void dequantize_kernel(float* output,
                                  const int8_t* input,
                                  const float* qscale,
                                  int output_size,
                                  int hidden_dim,
                                  int groups,
                                  int merge_count)
{
    unsigned merge_hidden = hidden_dim >> merge_count;
    unsigned quantization_stride = (merge_hidden * output_size) / groups;

    unsigned bid = blockIdx.x;
    unsigned tid = threadIdx.x;

    while (tid < output_size) {
        unsigned w_index = bid / merge_hidden;
        unsigned q_index = tid + bid * output_size;

        auto q = input[q_index];

        unsigned merge_hidden_total = w_index * merge_hidden;
        unsigned scale_index =
            ((((bid - merge_hidden_total) + tid * merge_hidden) / quantization_stride)
             << merge_count) +
            w_index;

        float scale_data = qscale[scale_index];

        output[q_index] = (scale_data * (float)q);
        tid += blockDim.x;
    }
}

__global__ void dequantize_kernel(__half* output,
                                  const int8_t* input,
                                  const float* qscale,
                                  unsigned output_size,
                                  unsigned hidden_dim,
                                  unsigned groups,
                                  unsigned merge_count)
{
#if __CUDA_ARCH__ >= 700

    unsigned merge_hidden = hidden_dim >> merge_count;
    unsigned quantization_stride = (merge_hidden * output_size) / groups;

    unsigned bid = blockIdx.x;
    unsigned tid = threadIdx.x;

    while (tid < output_size) {
        unsigned w_index = bid / merge_hidden;
        unsigned q_index = tid + bid * output_size;

        auto q = input[q_index];

        unsigned merge_hidden_total = w_index * merge_hidden;
        unsigned scale_index =
            ((((bid - merge_hidden_total) + tid * merge_hidden) / quantization_stride)
             << merge_count) +
            w_index;

        float scale_data = qscale[scale_index];

        output[q_index] = __float2half(scale_data * (float)q);
        tid += blockDim.x;
    }
#endif
}

template <typename T>
void launch_dequantize(T* output,
                       const int8_t* input,
                       const float* qscale,
                       unsigned output_size,
                       unsigned hidden_dim,
                       unsigned groups,
                       unsigned merge_count,
                       cudaStream_t stream)
{
    unsigned threads = 1024;
    dim3 block_dims(threads);
    dim3 grid_dims(hidden_dim);

    dequantize_kernel<<<grid_dims, block_dims, 0, stream>>>(
        output, input, qscale, output_size, hidden_dim, groups, merge_count);
}

template void launch_dequantize<float>(float*,
                                       const int8_t*,
                                       const float*,
                                       unsigned,
                                       unsigned,
                                       unsigned,
                                       unsigned,
                                       cudaStream_t);
template void launch_dequantize<__half>(__half*,
                                        const int8_t*,
                                        const float*,
                                        unsigned,
                                        unsigned,
                                        unsigned,
                                        unsigned,
                                        cudaStream_t);
