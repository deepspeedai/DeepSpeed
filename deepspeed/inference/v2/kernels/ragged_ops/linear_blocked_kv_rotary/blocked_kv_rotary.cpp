// Copyright (c) Microsoft Corporation.
// SPDX-License-Identifier: Apache-2.0

// DeepSpeed Team

#include "blocked_kv_rotary.h"
#include "ragged_kernel_helpers.h"

#define DISPATCH_KV_ROTARY(T_TYPE, C_TYPE)                                 \
    if (q.options().dtype() == torch::T_TYPE) {                            \
        launch_kv_rotary_kernel<C_TYPE>((C_TYPE*)kv_cache.data_ptr(),      \
                                        (C_TYPE*)q.data_ptr(),             \
                                        (C_TYPE*)k.data_ptr(),             \
                                        (C_TYPE*)v.data_ptr(),             \
                                        (C_TYPE*)inv_freq_ptr,             \
                                        rotary_dim,                        \
                                        theta_base,                        \
                                        batch_wrapper,                     \
                                        qkv_stride,                        \
                                        kv_cache_stride,                   \
                                        v_offset,                          \
                                        inv_freq_stride,                   \
                                        q_ratio,                           \
                                        head_size,                         \
                                        n_tokens,                          \
                                        n_q_heads,                         \
                                        at::cuda::getCurrentCUDAStream()); \
    }

/*
Rotary position embeddings + copy into KV cache. This implementation assumes
that the inverse frequencies should be ready from global memory rather than
synthesized in the kernel.

Arguments:
    kv_cache: [n_blocks, block_size, 2, n_kv_heads, head_size]
    q: [n_tokens, n_q_heads * head_size]
    k: [n_tokens, n_kv_heads * head_size]
    v: [n_tokens, n_kv_heads * head_size]
    inv_freq: [max_seq_len, head_size // 2]
*/
void kv_trained_rotary_embeddings(torch::Tensor& kv_cache,
                                  torch::Tensor& q,
                                  torch::Tensor& k,
                                  torch::Tensor& v,
                                  torch::Tensor& inv_freq,
                                  torch::Tensor& batch_metadata,
                                  torch::Tensor& seq_metadata,
                                  torch::Tensor& tokens_to_seq,
                                  torch::Tensor& kv_ptrs)
{
    const int32_t n_tokens = q.size(0);
    TORCH_CHECK(n_tokens == k.size(0));
    TORCH_CHECK(n_tokens == v.size(0));

    const float theta_base = 0.f;
    const int32_t rotary_dim = inv_freq.size(0) * 2;

    // Dimensions
    const int32_t block_size = kv_cache.size(1);
    const int32_t n_kv_heads = kv_cache.size(3);
    const int32_t head_size = kv_cache.size(4);

    // Strides
    const int32_t qkv_stride = q.stride(0);              // Per token
    const int32_t kv_cache_stride = kv_cache.stride(1);  // Per token
    const int32_t v_offset = kv_cache.stride(2);         // From k_cache to v_cache
    const int32_t inv_freq_stride = inv_freq.stride(0);  // Per token idx

    const int n_q_heads = q.size(1) / head_size;
    const int q_ratio = n_q_heads / n_kv_heads;

    void* inv_freq_ptr = (void*)inv_freq.data_ptr();

    BatchWrapperCPP batch_wrapper = make_cpp_batch_wrapper(
        batch_metadata, seq_metadata, tokens_to_seq, kv_ptrs, block_size, kv_cache.size(0));

    DISPATCH_KV_ROTARY(kHalf, __half);

#ifdef BF16_AVAILABLE
    DISPATCH_KV_ROTARY(kBFloat16, __nv_bfloat16);
#endif
}

/*
Rotary position embeddings + copy into KV cache. This implementation assumes
that the inverse frequencies should be synthesized in the kernel.

Arguments:
    kv_cache: [n_blocks, block_size, 2, n_kv_heads, head_size]
    q: [n_tokens, n_q_heads * head_size]
    k: [n_tokens, n_kv_heads * head_size]
    v: [n_tokens, n_kv_heads * head_size]
*/
void kv_rotary_embeddings(torch::Tensor& kv_cache,
                          torch::Tensor& q,
                          torch::Tensor& k,
                          torch::Tensor& v,
                          const int32_t rotary_dim,
                          const float theta_base,
                          torch::Tensor& batch_metadata,
                          torch::Tensor& seq_metadata,
                          torch::Tensor& tokens_to_seq,
                          torch::Tensor& kv_ptrs)
{
    const int32_t n_tokens = q.size(0);
    TORCH_CHECK(n_tokens == k.size(0));
    TORCH_CHECK(n_tokens == v.size(0));

    // Dimensions
    const int32_t block_size = kv_cache.size(1);
    const int32_t n_kv_heads = kv_cache.size(3);
    const int32_t head_size = kv_cache.size(4);

    // Strides
    const int32_t qkv_stride = q.stride(0);              // Per token
    const int32_t kv_cache_stride = kv_cache.stride(1);  // Per token
    const int32_t v_offset = kv_cache.stride(2);         // From k_cache to v_cache
    const int32_t inv_freq_stride = 0;                   // Per token idx

    const int n_q_heads = q.size(1) / head_size;
    const int q_ratio = n_q_heads / n_kv_heads;

    void* inv_freq_ptr = nullptr;

    BatchWrapperCPP batch_wrapper = make_cpp_batch_wrapper(
        batch_metadata, seq_metadata, tokens_to_seq, kv_ptrs, block_size, kv_cache.size(0));

    DISPATCH_KV_ROTARY(kHalf, __half);

#ifdef BF16_AVAILABLE
    DISPATCH_KV_ROTARY(kBFloat16, __nv_bfloat16);
#endif
}

#define DISPATCH_KV_COPY(T_TYPE, C_TYPE)                                 \
    if (q.options().dtype() == torch::T_TYPE) {                          \
        launch_kv_copy_kernel<C_TYPE>((C_TYPE*)kv_cache.data_ptr(),      \
                                      (C_TYPE*)q.data_ptr(),             \
                                      (C_TYPE*)k.data_ptr(),             \
                                      (C_TYPE*)v.data_ptr(),             \
                                      batch_wrapper,                     \
                                      qkv_stride,                        \
                                      kv_cache_stride,                   \
                                      v_offset,                          \
                                      q_ratio,                           \
                                      head_size,                         \
                                      n_tokens,                          \
                                      n_q_heads,                         \
                                      at::cuda::getCurrentCUDAStream()); \
    }

/*
Copy into linear KV cache.
*/
void linear_kv_copy(torch::Tensor& kv_cache,
                    torch::Tensor& q,
                    torch::Tensor& k,
                    torch::Tensor& v,
                    torch::Tensor& batch_metadata,
                    torch::Tensor& seq_metadata,
                    torch::Tensor& tokens_to_seq,
                    torch::Tensor& kv_ptrs)
{
    const int32_t n_tokens = q.size(0);
    TORCH_CHECK(n_tokens == k.size(0));
    TORCH_CHECK(n_tokens == v.size(0));

    // Dimensions
    const int32_t block_size = kv_cache.size(1);
    const int32_t n_kv_heads = kv_cache.size(3);
    const int32_t head_size = kv_cache.size(4);

    // Strides
    const int32_t qkv_stride = q.stride(0);  // Per token
    TORCH_CHECK(qkv_stride == k.stride(0));
    TORCH_CHECK(qkv_stride == v.stride(0));

    const int32_t kv_cache_stride = kv_cache.stride(1);  // Per token
    const int32_t v_offset = kv_cache.stride(2);         // From k_cache to v_cache

    const int n_q_heads = q.size(1) / head_size;

    TORCH_CHECK(n_q_heads % n_kv_heads == 0);
    const int q_ratio = n_q_heads / n_kv_heads;

    BatchWrapperCPP batch_wrapper = make_cpp_batch_wrapper(
        batch_metadata, seq_metadata, tokens_to_seq, kv_ptrs, block_size, kv_cache.size(0));

    DISPATCH_KV_COPY(kHalf, __half);

#ifdef BF16_AVAILABLE
    DISPATCH_KV_COPY(kBFloat16, __nv_bfloat16);
#endif
}
