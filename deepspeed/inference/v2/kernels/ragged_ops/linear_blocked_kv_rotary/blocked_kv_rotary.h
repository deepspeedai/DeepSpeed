// Copyright (c) Microsoft Corporation.
// SPDX-License-Identifier: Apache-2.0

// DeepSpeed Team

#pragma once

#include <c10/cuda/CUDAStream.h>
#include <torch/extension.h>
#include "blocked_kv_rotary.cuh"

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
                                  torch::Tensor& kv_ptrs);

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
                          torch::Tensor& kv_ptrs);

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
                    torch::Tensor& kv_ptrs);
