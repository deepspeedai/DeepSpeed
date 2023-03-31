// Copyright (c) Microsoft Corporation.
// SPDX-License-Identifier: Apache-2.0

// DeepSpeed Team

#include "custom_cuda_layers.h"
#include "memory_access_utils.h"

namespace cg = cooperative_groups;

template <typename T>
__global__ void slice_gpt_mask_impl(T* output_mask,
                                    const T* input_mask,
                                    int truncated_seq_len,
                                    int orig_seq_len)
{
    const int in_batch_stride = orig_seq_len * orig_seq_len;
    const int out_batch_stride = truncated_seq_len * truncated_seq_len;

    cg::thread_block tb = cg::this_thread_block();

    const T* input_mask_block =
        input_mask + blockIdx.x * in_batch_stride + blockIdx.y * orig_seq_len;
    T* output_mask_block =
        output_mask + blockIdx.x * out_batch_stride + blockIdx.y * truncated_seq_len;

    for (int i = tb.thread_index().x; i < truncated_seq_len; i += blockDim.x) {
        output_mask_block[i] = input_mask_block[i];
    }
}

template <typename T>
void launch_slice_gpt_mask(T* output_mask,
                           const T* input_mask,
                           int batch_size,
                           int truncated_seq_len,
                           int orig_seq_len,
                           cudaStream_t stream)
{
    const int threads = (truncated_seq_len >= 1024) ? 1024 : truncated_seq_len;

    dim3 block(threads);
    dim3 grid(batch_size, truncated_seq_len);

    slice_gpt_mask_impl<T>
        <<<grid, block, 0, stream>>>(output_mask, input_mask, truncated_seq_len, orig_seq_len);
}

template void launch_slice_gpt_mask<float>(float*, const float*, int, int, int, cudaStream_t);

template void launch_slice_gpt_mask<__half>(__half*, const __half*, int, int, int, cudaStream_t);

template <typename T>
__global__ void slice_bert_mask_impl(T* output_mask,
                                     const T* input_mask,
                                     const int32_t* retained_indices,
                                     int32_t truncated_seq_len,
                                     int32_t orig_seq_len)
{
    const int in_batch_stride = orig_seq_len * orig_seq_len;
    const int out_batch_stride = truncated_seq_len * truncated_seq_len;
    const int out_layer_stride = out_batch_stride * gridDim.y;

    cg::thread_block tb = cg::this_thread_block();

    const int out_layer_offset = tb.group_index().x * out_layer_stride;

    const int in_batch_offset = tb.group_index().y * in_batch_stride;
    const int out_batch_offset = tb.group_index().y * out_batch_stride;

    const int32_t gather_row =
        retained_indices[tb.group_index().y * truncated_seq_len + tb.group_index().z];
    const int in_seq_offset = gather_row * orig_seq_len;
    const int out_seq_offset = tb.group_index().z * truncated_seq_len;

    const T* in_sequence = input_mask + in_batch_offset + in_seq_offset;
    T* out_sequence = output_mask + out_layer_offset + out_batch_offset + out_seq_offset;
    const int32_t* gather_data = retained_indices + tb.group_index().y * truncated_seq_len;

    for (int i = tb.thread_index().x; i < truncated_seq_len; i += blockDim.x) {
        out_sequence[i] = in_sequence[gather_data[i]];
    }
}

/*
Since the Bert mask is not causal like GPT, we can't just generate a set of
masks for the entire model based off a single layer sample.

We map the kernel as follows:
z-dimension: layer
y-dimension: batch
x-dimension: sequence_offset
*/
template <typename T>
void launch_slice_bert_mask(T* output_mask,
                            const T* input_mask,
                            const int32_t* retained_indices,
                            int32_t layers,
                            int32_t batch_size,
                            int32_t truncated_seq_len,
                            int32_t orig_seq_len,
                            cudaStream_t stream)
{
    const int threads = (truncated_seq_len >= 1024) ? 1024 : truncated_seq_len;
    dim3 block(threads);
    dim3 grid(layers, batch_size, truncated_seq_len);

    slice_bert_mask_impl<T><<<grid, block, 0, stream>>>(
        output_mask, input_mask, retained_indices, truncated_seq_len, orig_seq_len);
}

template void launch_slice_bert_mask<float>(float*,
                                            const float*,
                                            const int32_t*,
                                            int32_t,
                                            int32_t,
                                            int32_t,
                                            int32_t,
                                            cudaStream_t);

template void launch_slice_bert_mask<__half>(__half*,
                                             const __half*,
                                             const int32_t*,
                                             int32_t,
                                             int32_t,
                                             int32_t,
                                             int32_t,
                                             cudaStream_t);
