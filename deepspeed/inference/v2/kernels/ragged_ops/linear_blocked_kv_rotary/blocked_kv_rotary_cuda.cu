// Copyright (c) Microsoft Corporation.
// SPDX-License-Identifier: Apache-2.0

// DeepSpeed Team

#include <cassert>
#include "blocked_kv_rotary.cuh"
#include "conversion_utils.h"
#include "ds_kernel_utils.h"
#include "memory_access_utils.h"

namespace cg = cooperative_groups;

namespace kv_rot {

constexpr int granularity = 16;
constexpr int threads = 256;

}  // namespace kv_rot

/*
Supports head size 32, 64, 128, 256
*/

template <typename T, int qRatio, int headSize, bool doRotary, int paddedHeadSize>
__global__ void kv_rotary_pos_kernel(T* kv_cache,
                                     T* q,
                                     T* k,
                                     T* v,
                                     const T* inv_freq,
                                     const int32_t rotary_dim,
                                     const float theta_base,
                                     const BatchWrapperCPP batch_desc,
                                     const int qkv_stride,
                                     const int kv_cache_stride,
                                     const int v_offset,
                                     const int inv_freq_stride)
{
    // Derived constexpr
    constexpr int vector_T = kv_rot::granularity / sizeof(T);
    constexpr int real_threads_per_head = headSize / vector_T;
    constexpr int threads_per_head = paddedHeadSize / vector_T;

    constexpr int tokens_per_block = kv_rot::threads / threads_per_head;

    // CG helpers
    cg::thread_block tb = cg::this_thread_block();
    cg::thread_block_tile<hw_warp_size> warp = cg::tiled_partition<hw_warp_size>(tb);
    cg::thread_block_tile<threads_per_head> head_group = cg::tiled_partition<threads_per_head>(tb);

    // Parallelize on the head dimension for X blocks
    const int head_idx = blockIdx.x;

    const int block_seq_idx = threadIdx.x / threads_per_head;
    const int base_neuron_idx = head_group.thread_rank() * vector_T;
    const int half_rotary_size = rotary_dim / 2;
    const int half_dim_lanes = half_rotary_size / vector_T;
    const int half_idx = base_neuron_idx % half_rotary_size;

    // Multiple tokens processed by the same threadblock
    const int token_idx = blockIdx.y * tokens_per_block + block_seq_idx;
    const bool valid_token = token_idx < batch_desc.batch_metadata->n_tokens;

    const bool valid_thread = valid_token && (head_group.thread_rank() < real_threads_per_head);
    const bool load_inv_freq = (inv_freq != nullptr) && valid_thread;

    // If we have GQA, then only one of the Q heads needs to do rotary + copy
    // for each of the heads in the group.
    bool need_kv = head_idx % qRatio == 0;
    // Make sure the following code is warp uniform
    need_kv = warp.shfl(need_kv, 0);

    const int kv_head_idx = head_idx / qRatio;

    // Ensure we don't access invalid portions of the seq_metadata
    const int32_t seq_id = (valid_thread) ? batch_desc.tokens_to_seq[token_idx] : 0;
    const InflightSeqDescriptor seq_desc = batch_desc.seq_metadata[seq_id];
    // This will give an invalid index if valid_thread is false, but should never affect memory.
    const int32_t global_token_idx = seq_desc.seen_tokens + (token_idx - seq_desc.start_idx);

    T* q_row = q + token_idx * qkv_stride + head_idx * headSize;
    T q_reg[vector_T];

    if (need_kv) {
        // The following logic assumes a linearly blocked KV cache. This means that no sparsity has
        // been introduced into cache history.
        const KVCacheDescriptor kv_desc = batch_desc.kv_desc;
        const int32_t seq_kv_block_idx = global_token_idx / kv_desc.block_size;
        const int32_t mapped_kv_block_idx =
            (valid_thread) ? kv_desc.block_lists[seq_id][seq_kv_block_idx] : 0;

        const int32_t kv_block_offset = global_token_idx % kv_desc.block_size;
        const int32_t kv_offset =
            (mapped_kv_block_idx * kv_desc.block_size + kv_block_offset) * kv_cache_stride +
            kv_head_idx * headSize;

        // Load indices from QKV output
        T* k_row = k + token_idx * qkv_stride + kv_head_idx * headSize;
        T* v_row = v + token_idx * qkv_stride + kv_head_idx * headSize;

        T k_reg[vector_T], v_reg[vector_T], inv_freq_reg[vector_T];

        mem_access::load_global<kv_rot::granularity>(q_reg, q_row + base_neuron_idx, valid_thread);
        mem_access::load_global<kv_rot::granularity>(k_reg, k_row + base_neuron_idx, valid_thread);
        mem_access::load_global<kv_rot::granularity>(v_reg, v_row + base_neuron_idx, valid_thread);
        mem_access::load_global<kv_rot::granularity>(
            inv_freq_reg, inv_freq + half_idx, load_inv_freq);
        if constexpr (doRotary) {
#pragma unroll
            for (int i = 0; i < vector_T; i++) {
                const int head_neuron_idx = base_neuron_idx + i;

                float inv_freq_flt;
                if (inv_freq != nullptr) {
                    inv_freq_flt = conversion::to<float>(inv_freq_reg[i]) * (float)global_token_idx;
                } else {
                    inv_freq_flt =
                        (float)((head_neuron_idx % half_rotary_size) * 2) / (float)rotary_dim;
                    // Conversion to T and back means that both branches of this if statement
                    // will produce the same results if using the same algo for producing the
                    // freqs.
                    T trunc_freq = conversion::to<T>(1.0 / powf(theta_base, inv_freq_flt));
                    inv_freq_flt = conversion::to<float>(trunc_freq) * (float)global_token_idx;
                }

                float rotary_sign = (head_neuron_idx >= half_rotary_size) ? -1.0f : 1.0f;
                float q_f = conversion::to<float>(q_reg[i]);
                float k_f = conversion::to<float>(k_reg[i]);
                float q_rot = q_f * rotary_sign;
                float k_rot = k_f * rotary_sign;

                const int target_lane = (head_neuron_idx < half_rotary_size)
                                            ? head_group.thread_rank() + half_dim_lanes
                                            : head_group.thread_rank() - half_dim_lanes;

                const float q_rot_temp = head_group.shfl(q_rot, target_lane);
                const float k_rot_temp = head_group.shfl(k_rot, target_lane);

                if (base_neuron_idx < rotary_dim) {
                    q_reg[i] = conversion::to<T>(q_f * cosf(inv_freq_flt) +
                                                 q_rot_temp * sinf(inv_freq_flt));
                    k_reg[i] = conversion::to<T>(k_f * cosf(inv_freq_flt) +
                                                 k_rot_temp * sinf(inv_freq_flt));
                }
            }
        }

        if (valid_thread) {
            mem_access::store_global<kv_rot::granularity>(kv_cache + kv_offset + base_neuron_idx,
                                                          k_reg);
            mem_access::store_global<kv_rot::granularity>(
                kv_cache + kv_offset + base_neuron_idx + v_offset, v_reg);
        }
    } else {
        T inv_freq_reg[vector_T];

        mem_access::load_global<kv_rot::granularity>(q_reg, q_row + base_neuron_idx, valid_thread);
        mem_access::load_global<kv_rot::granularity>(
            inv_freq_reg, inv_freq + half_idx, load_inv_freq);

        if constexpr (doRotary) {
#pragma unroll
            for (int i = 0; i < vector_T; i++) {
                const int head_neuron_idx = base_neuron_idx + i;

                float inv_freq_flt;
                if (inv_freq != nullptr) {
                    inv_freq_flt = conversion::to<float>(inv_freq_reg[i]) * (float)global_token_idx;
                } else {
                    inv_freq_flt =
                        (float)((head_neuron_idx % half_rotary_size) * 2) / (float)rotary_dim;
                    inv_freq_flt = 1.0 / powf(theta_base, inv_freq_flt) * (float)global_token_idx;
                }

                float rotary_sign = (head_neuron_idx >= half_rotary_size) ? -1.0f : 1.0f;
                float q_f = conversion::to<float>(q_reg[i]);
                float q_rot = q_f * rotary_sign;

                const int target_lane = (head_neuron_idx < half_rotary_size)
                                            ? head_group.thread_rank() + half_dim_lanes
                                            : head_group.thread_rank() - half_dim_lanes;

                const float q_rot_temp = head_group.shfl(q_rot, target_lane);
                if (base_neuron_idx < rotary_dim)
                    q_reg[i] = conversion::to<T>(q_f * cosf(inv_freq_flt) +
                                                 q_rot_temp * sinf(inv_freq_flt));
            }
        }
    }

    if (valid_thread && doRotary) {
        mem_access::store_global<kv_rot::granularity>(q_row + base_neuron_idx, q_reg);
    }
}

#define DISPATCH_KV_ROTARY_IMPL(Q_RATIO, HEAD_SIZE, PADDED_HEAD_SIZE)       \
    if (q_ratio == Q_RATIO && head_size == HEAD_SIZE)                       \
        kv_rotary_pos_kernel<T, Q_RATIO, HEAD_SIZE, true, PADDED_HEAD_SIZE> \
            <<<grid, block, 0, stream>>>(kv_cache,                          \
                                         q,                                 \
                                         k,                                 \
                                         v,                                 \
                                         inv_freq,                          \
                                         rotary_dim,                        \
                                         theta_base,                        \
                                         batch_desc,                        \
                                         qkv_stride,                        \
                                         kv_cache_stride,                   \
                                         v_offset,                          \
                                         inv_freq_stride);

#define LAUNCH_KV_ROTARY_FOR_Q_RATIO_HEAD_SIZE(Q_RATIO, HEAD_SIZE) \
    if (padded_head_size == 64) {                                  \
        DISPATCH_KV_ROTARY_IMPL(Q_RATIO, HEAD_SIZE, 64);           \
    } else if (padded_head_size == 128) {                          \
        DISPATCH_KV_ROTARY_IMPL(Q_RATIO, HEAD_SIZE, 128);          \
    } else {                                                       \
        assert(false);                                             \
    }

#define LAUNCH_KV_ROTARY_FOR_Q_RATIO(Q_RATIO)                 \
    if (head_size == 64) {                                    \
        LAUNCH_KV_ROTARY_FOR_Q_RATIO_HEAD_SIZE(Q_RATIO, 64);  \
    } else if (head_size == 80) {                             \
        LAUNCH_KV_ROTARY_FOR_Q_RATIO_HEAD_SIZE(Q_RATIO, 80);  \
    } else if (head_size == 128) {                            \
        LAUNCH_KV_ROTARY_FOR_Q_RATIO_HEAD_SIZE(Q_RATIO, 128); \
    } else {                                                  \
        assert(false);                                        \
    }

template <typename T>
void launch_kv_rotary_kernel(T* kv_cache,
                             T* q,
                             T* k,
                             T* v,
                             T* inv_freq,
                             const int32_t rotary_dim,
                             const float theta_base,
                             const BatchWrapperCPP batch_desc,
                             const int qkv_stride,
                             const int kv_cache_stride,
                             const int v_offset,
                             const int inv_freq_stride,
                             const int q_ratio,
                             const int head_size,
                             const int n_tokens,
                             const int n_q_heads,
                             cudaStream_t stream)
{
    constexpr int vector_T = kv_rot::granularity / sizeof(T);

    const int padded_head_size = next_pow2(head_size);
    const int threads_per_head = padded_head_size / vector_T;

    const int tokens_per_block = kv_rot::threads / threads_per_head;

    const dim3 block(kv_rot::threads);
    const int token_blocks = (n_tokens + tokens_per_block - 1) / tokens_per_block;
    const dim3 grid(n_q_heads, token_blocks);

    LAUNCH_KV_ROTARY_FOR_Q_RATIO(1)
    LAUNCH_KV_ROTARY_FOR_Q_RATIO(2)
    LAUNCH_KV_ROTARY_FOR_Q_RATIO(4)
    LAUNCH_KV_ROTARY_FOR_Q_RATIO(5)
    LAUNCH_KV_ROTARY_FOR_Q_RATIO(8)
    LAUNCH_KV_ROTARY_FOR_Q_RATIO(16)
    LAUNCH_KV_ROTARY_FOR_Q_RATIO(29)
    LAUNCH_KV_ROTARY_FOR_Q_RATIO(35)
    LAUNCH_KV_ROTARY_FOR_Q_RATIO(36)
    LAUNCH_KV_ROTARY_FOR_Q_RATIO(71)
}

#define INSTANTIATE_KV_ROTARY_KERNEL(TYPE)                                        \
    template void launch_kv_rotary_kernel<TYPE>(TYPE * kv_cache,                  \
                                                TYPE * q,                         \
                                                TYPE * k,                         \
                                                TYPE * v,                         \
                                                TYPE * inv_freq,                  \
                                                const int32_t rotary_dim,         \
                                                const float theta_base,           \
                                                const BatchWrapperCPP batch_desc, \
                                                const int qkv_stride,             \
                                                const int kv_cache_stride,        \
                                                const int v_offset,               \
                                                const int inv_freq_stride,        \
                                                const int q_ratio,                \
                                                const int head_size,              \
                                                const int n_tokens,               \
                                                const int n_q_heads,              \
                                                cudaStream_t stream);

INSTANTIATE_KV_ROTARY_KERNEL(__half)

#ifdef BF16_AVAILABLE
INSTANTIATE_KV_ROTARY_KERNEL(__nv_bfloat16)
#endif

#define DISPATCH_KV_COPY_IMPL(Q_RATIO, HEAD_SIZE, PADDED_HEAD_SIZE)          \
    if (q_ratio == Q_RATIO && head_size == HEAD_SIZE)                        \
        kv_rotary_pos_kernel<T, Q_RATIO, HEAD_SIZE, false, PADDED_HEAD_SIZE> \
            <<<grid, block, 0, stream>>>(kv_cache,                           \
                                         q,                                  \
                                         k,                                  \
                                         v,                                  \
                                         nullptr,                            \
                                         -1,                                 \
                                         0.f,                                \
                                         batch_desc,                         \
                                         qkv_stride,                         \
                                         kv_cache_stride,                    \
                                         v_offset,                           \
                                         0);

#define LAUNCH_KV_COPY_FOR_Q_RATIO_HEAD_SIZE(Q_RATIO, HEAD_SIZE) \
    if (padded_head_size == 64) {                                \
        DISPATCH_KV_COPY_IMPL(Q_RATIO, HEAD_SIZE, 64);           \
    } else if (padded_head_size == 128) {                        \
        DISPATCH_KV_COPY_IMPL(Q_RATIO, HEAD_SIZE, 128);          \
    } else {                                                     \
        assert(false);                                           \
    }

#define LAUNCH_KV_COPY_FOR_Q_RATIO(Q_RATIO)                 \
    if (head_size == 64) {                                  \
        LAUNCH_KV_COPY_FOR_Q_RATIO_HEAD_SIZE(Q_RATIO, 64);  \
    } else if (head_size == 80) {                           \
        LAUNCH_KV_COPY_FOR_Q_RATIO_HEAD_SIZE(Q_RATIO, 80);  \
    } else if (head_size == 128) {                          \
        LAUNCH_KV_COPY_FOR_Q_RATIO_HEAD_SIZE(Q_RATIO, 128); \
    } else {                                                \
        assert(false);                                      \
    }

template <typename T>
void launch_kv_copy_kernel(T* kv_cache,
                           T* q,
                           T* k,
                           T* v,
                           const BatchWrapperCPP batch_desc,
                           const int qkv_stride,
                           const int kv_cache_stride,
                           const int v_offset,
                           const int q_ratio,
                           const int head_size,
                           const int n_tokens,
                           const int n_q_heads,
                           cudaStream_t stream)
{
    constexpr int vector_T = kv_rot::granularity / sizeof(T);
    const int padded_head_size = next_pow2(head_size);
    const int threads_per_head = padded_head_size / vector_T;
    const int tokens_per_block = kv_rot::threads / threads_per_head;

    const dim3 block(kv_rot::threads);
    const int token_blocks = (n_tokens + tokens_per_block - 1) / tokens_per_block;
    const dim3 grid(n_q_heads, token_blocks);

    LAUNCH_KV_COPY_FOR_Q_RATIO(1)
    LAUNCH_KV_COPY_FOR_Q_RATIO(2)
    LAUNCH_KV_COPY_FOR_Q_RATIO(4)
    LAUNCH_KV_COPY_FOR_Q_RATIO(5)
    LAUNCH_KV_COPY_FOR_Q_RATIO(8)
}

#define INSTANTIATE_KV_COPY_KERNEL(TYPE)                                        \
    template void launch_kv_copy_kernel<TYPE>(TYPE * kv_cache,                  \
                                              TYPE * q,                         \
                                              TYPE * k,                         \
                                              TYPE * v,                         \
                                              const BatchWrapperCPP batch_desc, \
                                              const int qkv_stride,             \
                                              const int kv_cache_stride,        \
                                              const int v_offset,               \
                                              const int q_ratio,                \
                                              const int head_size,              \
                                              const int n_tokens,               \
                                              const int n_q_heads,              \
                                              cudaStream_t stream);

INSTANTIATE_KV_COPY_KERNEL(__half)

#ifdef BF16_AVAILABLE
INSTANTIATE_KV_COPY_KERNEL(__nv_bfloat16)
#endif
