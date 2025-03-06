// Copyright (c) Microsoft Corporation.
// SPDX-License-Identifier: Apache-2.0

// DeepSpeed Team

#include "ds_kernel_utils.h"
#include "logits_gather.cuh"
#include "memory_access_utils.h"
#include "ragged_dtypes.h"

namespace logits_gather {

constexpr int granularity = 16;
constexpr int threads = 512;

}  // namespace logits_gather

template <typename T>
__global__ void logits_gather_kernel(T* final_token_acts,
                                     const T* token_acts,
                                     const RaggedBatchDescriptor* ragged_batch,
                                     const InflightSeqDescriptor* inflight_batch,
                                     const int32_t embed_dim)
{
    constexpr int T_vector = logits_gather::granularity / sizeof(T);

    const int32_t seq_id = blockIdx.y;

    // It's possible we've padded the output Tensor (under CG conditions)
    if (seq_id >= ragged_batch->n_sequences) return;

    const InflightSeqDescriptor seq = inflight_batch[seq_id];
    const int final_token_idx = seq.start_idx + seq.n_tokens - 1;

    const int token_offset = final_token_idx * embed_dim;
    const int thread_offset =
        threadIdx.x * T_vector + blockIdx.x * logits_gather::threads * T_vector;

    const int final_token_offset = seq_id * embed_dim;

    T reg_buf[T_vector];

    if (thread_offset < embed_dim) {
        mem_access::load_global<logits_gather::granularity>(
            reg_buf, token_acts + token_offset + thread_offset);

        mem_access::store_global<logits_gather::granularity>(
            final_token_acts + final_token_offset + thread_offset, reg_buf);
    }
}

template <typename T>
void launch_logits_gather(T* final_token_acts,
                          const T* all_acts,
                          const RaggedBatchDescriptor* ragged_batch,
                          const InflightSeqDescriptor* inflight_batch,
                          const int32_t n_seqs,
                          const int32_t embed_dim,
                          cudaStream_t stream)
{
    constexpr int T_vector = logits_gather::granularity / sizeof(T);
    constexpr int elems_per_block = logits_gather::threads * T_vector;
    const int parallel_blocks = (embed_dim + elems_per_block - 1) / elems_per_block;

    const dim3 grid(parallel_blocks, n_seqs, 1);
    const dim3 block(logits_gather::threads, 1, 1);

    logits_gather_kernel<T><<<grid, block, 0, stream>>>(
        final_token_acts, all_acts, ragged_batch, inflight_batch, embed_dim);
}

#define INSTANTIATE_FOR_TYPE(T)                                                        \
    template void launch_logits_gather<T>(T * final_token_acts,                        \
                                          const T* all_acts,                           \
                                          const RaggedBatchDescriptor* ragged_batch,   \
                                          const InflightSeqDescriptor* inflight_batch, \
                                          const int32_t n_seqs,                        \
                                          const int32_t embed_dim,                     \
                                          cudaStream_t stream);

INSTANTIATE_FOR_TYPE(float)
INSTANTIATE_FOR_TYPE(__half)

#ifdef BF16_AVAILABLE
INSTANTIATE_FOR_TYPE(__nv_bfloat16)
#endif
