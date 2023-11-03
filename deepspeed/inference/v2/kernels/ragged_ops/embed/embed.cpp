// Copyright (c) Microsoft Corporation.
// SPDX-License-Identifier: Apache-2.0

// DeepSpeed Team

#include "embed.h"
#include "ragged_kernel_helpers.h"

#ifdef BF16_AVAILABLE
#define DISPATCH_FOR_FLOAT(DTYPE, ...)                       \
    [&] {                                                    \
        if (DTYPE == torch::kFloat32) {                      \
            using float_t = float;                           \
            return __VA_ARGS__();                            \
        } else if (DTYPE == torch::kFloat16) {               \
            using float_t = __half;                          \
            return __VA_ARGS__();                            \
        } else if (DTYPE == torch::kBFloat16) {              \
            using float_t = __nv_bfloat16;                   \
            return __VA_ARGS__();                            \
        } else {                                             \
            TORCH_CHECK(false, "Unsupported dispatch type"); \
        }                                                    \
    }()
#else
#define DISPATCH_FOR_FLOAT(DTYPE, ...)                       \
    [&] {                                                    \
        if (DTYPE == torch::kFloat32) {                      \
            using float_t = float;                           \
            return __VA_ARGS__();                            \
        } else if (DTYPE == torch::kFloat16) {               \
            using float_t = __half;                          \
            return __VA_ARGS__();                            \
        } else {                                             \
            TORCH_CHECK(false, "Unsupported dispatch type"); \
        }                                                    \
    }()
#endif

#define DISPATCH_FOR_INT(DTYPE, ...)                         \
    [&] {                                                    \
        if (DTYPE == torch::kInt32) {                        \
            using int_t = int32_t;                           \
            return __VA_ARGS__();                            \
        } else if (DTYPE == torch::kInt64) {                 \
            using int_t = int64_t;                           \
            return __VA_ARGS__();                            \
        } else {                                             \
            TORCH_CHECK(false, "Unsupported dispatch type"); \
        }                                                    \
    }()

/*
Embeddings kernel aware of ragged batch structure.
*/
void ragged_embed(torch::Tensor& embedded_tokens,
                  torch::Tensor& input_ids,
                  torch::Tensor& embedding_weight,
                  c10::optional<torch::Tensor>& position_embedding_weight,
                  int32_t pos_embed_offset,
                  torch::Tensor& batch_metadata,
                  torch::Tensor& seq_metadata,
                  torch::Tensor& tokens_to_seq,
                  torch::Tensor& kv_ptrs)
{
    // We don't care about KV cache here, so just hardcoding 0s for block_size/num_blocks
    BatchWrapperCPP batch_wrapper =
        make_cpp_batch_wrapper(batch_metadata, seq_metadata, tokens_to_seq, kv_ptrs, 0, 0);

    const int32_t n_tokens = input_ids.numel();
    const int32_t embed_dim = embedding_weight.size(1);
    const int32_t vocab_size = embedding_weight.size(0);

    DISPATCH_FOR_INT(input_ids.scalar_type(), [&] {
        DISPATCH_FOR_FLOAT(embedding_weight.scalar_type(), [&] {
            float_t* pos_embed_ptr = nullptr;
            int32_t max_position_embed_idx = 0;
            if (position_embedding_weight.has_value()) {
                TORCH_CHECK(
                    position_embedding_weight.value().options().dtype() ==
                        embedding_weight.options().dtype(),
                    "position_embedding_weight and embedding_weight must have the same dtype");
                pos_embed_ptr =
                    reinterpret_cast<float_t*>(position_embedding_weight.value().data_ptr());
                max_position_embed_idx = position_embedding_weight.value().size(0) - 1;
            }

            launch_ragged_embed_kernel((float_t*)embedded_tokens.data_ptr(),
                                       (const int_t*)input_ids.data_ptr(),
                                       (const float_t*)embedding_weight.data_ptr(),
                                       pos_embed_ptr,
                                       batch_wrapper,
                                       n_tokens,
                                       embed_dim,
                                       vocab_size,
                                       max_position_embed_idx,
                                       pos_embed_offset,
                                       at::cuda::getCurrentCUDAStream());
        });
    });
}
