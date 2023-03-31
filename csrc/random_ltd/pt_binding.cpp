// Copyright (c) Microsoft Corporation.
// SPDX-License-Identifier: Apache-2.0

// DeepSpeed Team

#include <torch/extension.h>
#include <vector>
#include "custom_cuda_layers.h"

torch::Tensor token_sort_(torch::Tensor& unsorted_token_ids, int64_t original_tokens)
{
    const int layers = unsorted_token_ids.size(0);
    const int batch_size = unsorted_token_ids.size(1);
    const int reserved_tokens = unsorted_token_ids.size(2);

    launch_token_sort(unsorted_token_ids.data_ptr<int32_t>(),
                      layers,
                      batch_size,
                      reserved_tokens,
                      original_tokens,
                      c10::cuda::getCurrentCUDAStream());

    return unsorted_token_ids;
}

torch::Tensor token_gather(torch::Tensor& activations,
                           torch::Tensor& sorted_indices,
                           bool batch_first)
{
    // Activations may be in either [N, S, C] or [S, N, C] while sorted_indices is
    // always in [N, retained]
    /*
        TORCH_CHECK(sorted_indices.size(0) == activations.size(0) ||
                        sorted_indices.size(0) == activations.size(1),
                    "Unable to match the batch size of the sorted indices to the activation
       shape."); TORCH_CHECK(activations.size(2) % 8 == 0, "Channels must be divisible by 8 to align
       with vectorized loads.");
    */
    // bool batch_first = sorted_indices.size(0) == activations.size(0);

    const int64_t dim_0 = (batch_first) ? sorted_indices.size(0) : sorted_indices.size(1);
    const int64_t dim_1 = (batch_first) ? sorted_indices.size(1) : sorted_indices.size(0);
    const int64_t dim_2 = activations.size(2);

    auto output = torch::empty({dim_0, dim_1, dim_2}, activations.options());

    const int batch_size = sorted_indices.size(0);
    const int channels = dim_2;
    const int retained_tokens = sorted_indices.size(1);
    const int read_batch_stride = (batch_first) ? activations.stride(0) : activations.stride(1);
    const int read_seq_stride = (batch_first) ? activations.stride(1) : activations.stride(0);
    const int write_batch_stride = (batch_first) ? output.stride(0) : output.stride(1);
    const int write_seq_stride = (batch_first) ? output.stride(1) : output.stride(0);

    if (activations.options().dtype() == torch::kFloat) {
        launch_gather_tokens((float*)output.data_ptr(),
                             (float*)activations.data_ptr(),
                             (int32_t*)sorted_indices.data_ptr(),
                             batch_size,
                             retained_tokens,
                             channels,
                             read_batch_stride,
                             read_seq_stride,
                             write_batch_stride,
                             write_seq_stride,
                             c10::cuda::getCurrentCUDAStream());
    } else {
        launch_gather_tokens((__half*)output.data_ptr(),
                             (__half*)activations.data_ptr(),
                             (int32_t*)sorted_indices.data_ptr(),
                             batch_size,
                             retained_tokens,
                             channels,
                             read_batch_stride,
                             read_seq_stride,
                             write_batch_stride,
                             write_seq_stride,
                             c10::cuda::getCurrentCUDAStream());
    }

    return output;
}

torch::Tensor token_scatter_(torch::Tensor& all_activations,
                             torch::Tensor& layer_activations,
                             torch::Tensor& sorted_indices,
                             bool batch_first)
{
    // Activations may be in either [N, S, C] or [S, N, C] while sorted_indices is
    // always in [N, retained]
    /*
        TORCH_CHECK(sorted_indices.size(0) == all_activations.size(0) ||
                        sorted_indices.size(0) == all_activations.size(1),
                    "Unable to match the batch size of the sorted indices to the activation
       shape."); TORCH_CHECK(all_activations.size(2) % 8 != 0, "Channels must be divisible by 8 to
       align with vectorized loads.");
    */
    // bool batch_first = sorted_indices.size(0) == all_activations.size(0);

    const int batch_size = sorted_indices.size(0);
    const int channels = all_activations.size(2);
    const int retained_tokens = sorted_indices.size(1);
    const int read_batch_stride = (batch_first) ? layer_activations.stride(0)
                                                : layer_activations.stride(1);
    const int read_seq_stride = (batch_first) ? layer_activations.stride(1)
                                              : layer_activations.stride(0);
    const int write_batch_stride = (batch_first) ? all_activations.stride(0)
                                                 : all_activations.stride(1);
    const int write_seq_stride = (batch_first) ? all_activations.stride(1)
                                               : all_activations.stride(0);

    if (all_activations.options().dtype() == torch::kFloat) {
        launch_scatter_tokens((float*)all_activations.data_ptr(),
                              (float*)layer_activations.data_ptr(),
                              (int32_t*)sorted_indices.data_ptr(),
                              batch_size,
                              retained_tokens,
                              channels,
                              read_batch_stride,
                              read_seq_stride,
                              write_batch_stride,
                              write_seq_stride,
                              c10::cuda::getCurrentCUDAStream());
    } else {
        launch_scatter_tokens((__half*)all_activations.data_ptr(),
                              (__half*)layer_activations.data_ptr(),
                              (int32_t*)sorted_indices.data_ptr(),
                              batch_size,
                              retained_tokens,
                              channels,
                              read_batch_stride,
                              read_seq_stride,
                              write_batch_stride,
                              write_seq_stride,
                              c10::cuda::getCurrentCUDAStream());
    }

    return all_activations;
}

torch::Tensor mask_gather_bert(torch::Tensor& dense_mask, torch::Tensor& sorted_indices)
{
    // TORCH_CHECK(dense_mask.dim() == 4)

    const int batch_size = dense_mask.size(0);
    const int layers = sorted_indices.size(0);
    /*
        TORCH_CHECK(layers * batch_size == sorted_indices.size(0),
                    "Mismatch between the indices and the mask");
    */
    const int orig_seq_len = dense_mask.size(3);
    const int truncated_seq_len = sorted_indices.size(2);

    auto output = torch::empty({layers, batch_size, 1, truncated_seq_len, truncated_seq_len},
                               dense_mask.options());

    if (dense_mask.options().dtype() == torch::kFloat) {
        launch_slice_bert_mask((float*)output.data_ptr(),
                               (const float*)dense_mask.data_ptr(),
                               (const int32_t*)sorted_indices.data_ptr(),
                               layers,
                               batch_size,
                               truncated_seq_len,
                               orig_seq_len,
                               c10::cuda::getCurrentCUDAStream());
    } else {
        launch_slice_bert_mask((__half*)output.data_ptr(),
                               (const __half*)dense_mask.data_ptr(),
                               (const int32_t*)sorted_indices.data_ptr(),
                               layers,
                               batch_size,
                               truncated_seq_len,
                               orig_seq_len,
                               c10::cuda::getCurrentCUDAStream());
    }

    return output;
}

torch::Tensor mask_gather_gpt(torch::Tensor dense_mask, int truncated_seq_len)
{
    // TORCH_CHECK(dense_mask.dim() == 4)

    const int batch_size = dense_mask.size(0);
    const int orig_seq_len = dense_mask.size(3);

    auto output =
        torch::empty({batch_size, 1, truncated_seq_len, truncated_seq_len}, dense_mask.options());

    if (dense_mask.options().dtype() == torch::kFloat) {
        launch_slice_gpt_mask((float*)output.data_ptr(),
                              (const float*)dense_mask.data_ptr(),
                              batch_size,
                              truncated_seq_len,
                              orig_seq_len,
                              c10::cuda::getCurrentCUDAStream());
    } else {
        launch_slice_gpt_mask((__half*)output.data_ptr(),
                              (const __half*)dense_mask.data_ptr(),
                              batch_size,
                              truncated_seq_len,
                              orig_seq_len,
                              c10::cuda::getCurrentCUDAStream());
    }

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("token_sort_", &token_sort_, "Comparison free sorting algorithm (CUDA)");
    m.def("token_gather", &token_gather, "Parallel gather of tokens (CUDA)");
    m.def("token_scatter_", &token_scatter_, "Parallel scatter of tokens (CUDA)");
    m.def("mask_gather_bert", &mask_gather_bert, "Token-based mask gather for BERT masking (CUDA)");
    m.def("mask_gather_gpt", &mask_gather_gpt, "Token-based mask gather for GPT masking (CUDA)");
}
