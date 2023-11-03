// Copyright (c) Microsoft Corporation.
// SPDX-License-Identifier: Apache-2.0

// DeepSpeed Team

#include "conversion_utils.h"
#include "memory_access_utils.h"
#include "reduction_utils.h"
#include "top_1_gating.cuh"

using ROp = reduce::ROpType;

template <typename T>
__global__ void top_1_gating_kernel(int32_t* expert_counts,
                                    float* scores,
                                    int32_t* assignments,
                                    int32_t* offsets,
                                    const T* logits,
                                    const RaggedBatchDescriptor* batch_metadata,
                                    const int32_t n_experts)
{
    const int32_t token_idx = blockIdx.x;
    const int32_t expert_idx = threadIdx.x;
    const int32_t max_warps = 1024 / hw_warp_size;

    // CG helpers
    cg::thread_block tb = cg::this_thread_block();
    cg::thread_block_tile<hw_warp_size> warp = cg::tiled_partition<hw_warp_size>(tb);

    // Padding tokens do not require
    if (token_idx >= batch_metadata->n_tokens) {
        if (threadIdx.x == 0) {
            offsets[token_idx] = gating::unassigned;
            assignments[token_idx] = gating::unassigned;
        }
        return;
    }

    const T* token_logits = logits + token_idx * n_experts;

    float logit_val;
    if (expert_idx < n_experts) {
        logit_val = conversion::to<float>(token_logits[expert_idx]);
    } else {
        reduce::init<ROp::Max>(&logit_val);
    }

    // Training code tends to use ``torch.argmax`` to select the expert, which
    // which has ties broken by the lower index. Since our fused comparison algorithm
    // breaks ties by the higher index (since it's the lower 32-bits of the 64-bit
    // comparison), we invert the expert index to break ties by the lower index.
    int32_t inverted_expert = n_experts - expert_idx - 1;
    // Perform softmax
    const reduce::IdxReduceResult res =
        reduce::idx_reduce<ROp::Max, max_warps>(tb, warp, logit_val, inverted_expert);
    // Recover the original expert index
    const int32_t assigned_expert = n_experts - res.idx - 1;
    const float max_logit = res.val;

    float softmax_sum = __expf(logit_val - max_logit);
    reduce::block<ROp::Add>(tb, warp, softmax_sum);

    // Compute the score
    const float score = __expf(max_logit - max_logit) / softmax_sum;

    if (threadIdx.x == 0) {
        scores[token_idx] = score;
        assignments[token_idx] = assigned_expert;
        offsets[token_idx] = atomicAdd(expert_counts + assigned_expert, 1);
    }
}

template <typename T>
void launch_top_1_gating(int32_t* expert_counts,
                         float* scores,
                         int32_t* assignments,
                         int32_t* offsets,
                         const T* logits,
                         const RaggedBatchDescriptor* batch_metadata,
                         const int32_t n_tokens,
                         const int32_t n_experts,
                         cudaStream_t stream)
{
    const dim3 grid(n_tokens);
    const dim3 block(((n_experts + hw_warp_size - 1) / hw_warp_size) * hw_warp_size);

    top_1_gating_kernel<T><<<grid, block, 0, stream>>>(
        expert_counts, scores, assignments, offsets, logits, batch_metadata, n_experts);
}

#define INSTANTIATE_TOP_1_KERNEL(T)                                                   \
    template void launch_top_1_gating<T>(int32_t * expert_counts,                     \
                                         float* scores,                               \
                                         int32_t* assignments,                        \
                                         int32_t* offsets,                            \
                                         const T* logits,                             \
                                         const RaggedBatchDescriptor* batch_metadata, \
                                         const int32_t n_tokens,                      \
                                         const int32_t n_experts,                     \
                                         cudaStream_t stream);

INSTANTIATE_TOP_1_KERNEL(float)
INSTANTIATE_TOP_1_KERNEL(__half)
#ifdef BF16_AVAILABLE
INSTANTIATE_TOP_1_KERNEL(__nv_bfloat16)
#endif
