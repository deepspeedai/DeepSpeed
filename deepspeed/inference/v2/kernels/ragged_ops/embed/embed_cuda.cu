// Copyright (c) Microsoft Corporation.
// SPDX-License-Identifier: Apache-2.0

// DeepSpeed Team

#include "ds_kernel_utils.h"
#include "embed.cuh"
#include "memory_access_utils.h"
#include "ragged_dtypes.h"

namespace embed {

constexpr int granularity = 16;
constexpr int threads = 512;

}  // namespace embed

template <typename TokenType, typename EmbedType>
__global__ void ragged_embed_kernel(EmbedType* embedded_tokens,
                                    const TokenType* input_ids,
                                    const EmbedType* embedding_weight,
                                    const EmbedType* position_weight,
                                    const BatchWrapperCPP batch_desc,
                                    const int32_t embed_dim,
                                    const int32_t vocab_size,
                                    const int32_t max_position_embed_idx,
                                    const int32_t position_embed_offset)
{
    constexpr int T_vector = embed::granularity / sizeof(EmbedType);

    const int32_t token_idx = blockIdx.y;

    // It's possible our batch is padded (under CG conditions typically)
    if (token_idx >= batch_desc.batch_metadata->n_tokens) return;

    TokenType token_value = input_ids[token_idx];

    if (token_value >= vocab_size || token_value < 0) {
        // TODO(cmikeh2): This is invalid, but not sure how we want to handle it being invalid
        // yet.
        return;
    }

    const EmbedType* embedding_row = embedding_weight + token_value * embed_dim;
    EmbedType* dest_row = embedded_tokens + token_idx * embed_dim;

    const int channel_offset = (threadIdx.x + embed::threads * blockIdx.x) * T_vector;

    if (channel_offset < embed_dim) {
        EmbedType reg_buf[T_vector];

        mem_access::load_global<embed::granularity>(reg_buf, embedding_row + channel_offset);

        if (position_weight != nullptr) {
            // Map the token to its global idx (indirect memory accesses aren't great but whatever)
            const int32_t seq_idx = batch_desc.tokens_to_seq[token_idx];
            const InflightSeqDescriptor seq_desc = batch_desc.seq_metadata[seq_idx];
            int32_t pos_emb_idx = seq_desc.seen_tokens + (token_idx - seq_desc.start_idx);

            // Position embed offset is an OPT-specific feature I think?
            pos_emb_idx = pos_emb_idx + position_embed_offset;

            // This clamping is technically
            pos_emb_idx = (pos_emb_idx < 0) ? 0 : pos_emb_idx;
            pos_emb_idx = (pos_emb_idx >= max_position_embed_idx) ? max_position_embed_idx
                                                                  : pos_emb_idx;

            const EmbedType* position_embedding_row = position_weight + pos_emb_idx * embed_dim;

            EmbedType pos_buf[T_vector];
            mem_access::load_global<embed::granularity>(pos_buf,
                                                        position_embedding_row + channel_offset);

#pragma unroll
            for (int i = 0; i < T_vector; i++) { reg_buf[i] += pos_buf[i]; }
        }

        mem_access::store_global<embed::granularity>(dest_row + channel_offset, reg_buf);
    }
}

template <typename TokenType, typename EmbedType>
void launch_ragged_embed_kernel(EmbedType* embedded_tokens,
                                const TokenType* input_ids,
                                const EmbedType* embedding_weight,
                                const EmbedType* position_weight,
                                const BatchWrapperCPP batch_desc,
                                const int32_t n_tokens,
                                const int32_t embed_dim,
                                const int32_t vocab_size,
                                const int32_t max_position_embed_idx,
                                const int32_t position_embed_offset,
                                cudaStream_t stream)
{
    constexpr int T_vector = embed::granularity / sizeof(EmbedType);
    constexpr int elems_per_block = embed::threads * T_vector;
    const int parallel_blocks = (embed_dim + elems_per_block - 1) / elems_per_block;

    const dim3 grid_dim(parallel_blocks, n_tokens, 1);
    const dim3 block_dim(embed::threads, 1, 1);

    ragged_embed_kernel<TokenType, EmbedType>
        <<<grid_dim, block_dim, 0, stream>>>(embedded_tokens,
                                             input_ids,
                                             embedding_weight,
                                             position_weight,
                                             batch_desc,
                                             embed_dim,
                                             vocab_size,
                                             max_position_embed_idx,
                                             position_embed_offset);
}

#define INSTANTIATE_EMBED_FOR_TYPES(TOKEN_TYPE, EMBED_TYPE)           \
    template void launch_ragged_embed_kernel<TOKEN_TYPE, EMBED_TYPE>( \
        EMBED_TYPE * embedded_tokens,                                 \
        const TOKEN_TYPE* input_ids,                                  \
        const EMBED_TYPE* embedding_weight,                           \
        const EMBED_TYPE* position_weight,                            \
        const BatchWrapperCPP batch_descriptor,                       \
        const int32_t n_tokens,                                       \
        const int32_t embed_dim,                                      \
        const int32_t vocab_size,                                     \
        const int32_t max_position_embed_idx,                         \
        const int32_t position_embed_offset,                          \
        cudaStream_t stream);

INSTANTIATE_EMBED_FOR_TYPES(int32_t, float)
INSTANTIATE_EMBED_FOR_TYPES(int64_t, float)

INSTANTIATE_EMBED_FOR_TYPES(int32_t, __half)
INSTANTIATE_EMBED_FOR_TYPES(int64_t, __half)

#ifdef BF16_AVAILABLE
INSTANTIATE_EMBED_FOR_TYPES(int32_t, __nv_bfloat16)
INSTANTIATE_EMBED_FOR_TYPES(int64_t, __nv_bfloat16)
#endif
