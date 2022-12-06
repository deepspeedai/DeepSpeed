/*
Copyright 2022 The Microsoft DeepSpeed Team
*/

#include "inference_cuda_layers.h"

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

__global__ void dequantize_kernel(float* output,
                                  const int8_t* input,
                                  const float* qscale,
                                  int hidden_dim,
                                  unsigned merge_hidden,
                                  int cnt)
{
}

__global__ void dequantize_kernel(__half* output,
                                  const int8_t* input,
                                  const float* qscale,
                                  unsigned hidden_dim,
                                  unsigned merge_hidden,
                                  int cnt)
{
    unsigned bid = blockIdx.x * gridDim.y + blockIdx.y;
    unsigned tid = threadIdx.x;

    float local_scale = qscale[blockIdx.x];

    const float* input_cast = reinterpret_cast<const float*>(input);
    float2* output_cast = reinterpret_cast<float2*>(output);

    input_cast += bid * merge_hidden;
    output_cast += bid * merge_hidden;

    for (int c = 0; c < cnt; c++) {
        if (tid < merge_hidden) {
            float q = input_cast[tid];
            int8_t* q_int8 = (int8_t*)&q;

            float2 q_f;
            __half* q_h = (__half*)&q_f;

            q_h[0] = __float2half(local_scale * (float)q_int8[0]);
            q_h[1] = __float2half(local_scale * (float)q_int8[1]);
            q_h[2] = __float2half(local_scale * (float)q_int8[2]);
            q_h[3] = __float2half(local_scale * (float)q_int8[3]);
            output_cast[tid] = q_f;
            tid += blockDim.x;
        }
    }
}

template <typename T>
void launch_dequantize(T* output,
                       const int8_t* input,
                       const float* qscale,
                       unsigned output_size,
                       unsigned hidden_dim,
                       unsigned groups,
                       cudaStream_t stream)
{
    unsigned threads = 1024;
    hidden_dim /= 4;
    unsigned hid_cnt = threads / hidden_dim;
    unsigned thd_cnt = (hidden_dim - 1) / threads + 1;
    hid_cnt = hid_cnt > 0 ? hid_cnt : 1;

    unsigned blocks = (output_size + hid_cnt * groups - 1) / (hid_cnt * groups);
    dim3 block_dims(threads);
    dim3 grid_dims(groups, blocks);

    dequantize_kernel<<<grid_dims, block_dims, 0, stream>>>(
        output, input, qscale, hidden_dim, hid_cnt * hidden_dim, thd_cnt);
}

template void launch_dequantize<float>(float*,
                                       const int8_t*,
                                       const float*,
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
                                        cudaStream_t);
