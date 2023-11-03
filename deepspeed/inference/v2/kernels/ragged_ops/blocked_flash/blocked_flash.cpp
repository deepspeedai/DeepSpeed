// Copyright (c) Microsoft Corporation.
// SPDX-License-Identifier: Apache-2.0

// DeepSpeed Team

/******************************************************************************
 * Copyright (c) 2023, Tri Dao.
 ******************************************************************************/

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>

#include "blocked_flash.h"
#include "flash.h"

#define CHECK_SHAPE(x, ...)                                     \
    TORCH_CHECK(x.sizes() == torch::IntArrayRef({__VA_ARGS__}), \
                #x " must have shape (" #__VA_ARGS__ ")")

void flash_attn_by_atoms(at::Tensor& out,
                         at::Tensor& q,
                         at::Tensor& k,
                         at::Tensor& v,
                         at::Tensor& attention_atoms,
                         const float softmax_scale,
                         const bool is_causal)
{
    auto dprops = at::cuda::getCurrentDeviceProperties();

    bool is_sm8x = dprops->major == 8 && dprops->minor >= 0;
    bool is_sm90 = dprops->major == 9 && dprops->minor == 0;
    TORCH_CHECK(is_sm90 || is_sm8x, "FlashAttention only supports Ampere GPUs or newer.");

    auto q_dtype = q.dtype();
    TORCH_CHECK(q_dtype == torch::kFloat16 || q_dtype == torch::kBFloat16,
                "FlashAttention only support fp16 and bf16 data type");
    if (q_dtype == torch::kBFloat16) {
        TORCH_CHECK(is_sm90 || is_sm8x, "bfloat16 is only supported on Ampere GPUs or newer");
    }
    TORCH_CHECK(k.dtype() == q_dtype, "query and key must have the same dtype");
    TORCH_CHECK(v.dtype() == q_dtype, "query and value must have the same dtype");

    TORCH_CHECK(q.is_cuda(), "Input tensor must be on CUDA device");
    TORCH_CHECK(k.is_cuda(), "Input tensor must be on CUDA device");
    TORCH_CHECK(v.is_cuda(), "Input tensor must be on CUDA device");

    TORCH_CHECK(q.stride(-1) == 1, "Input tensor must have contiguous last dimension");
    TORCH_CHECK(k.stride(-1) == 1, "Input tensor must have contiguous last dimension");
    TORCH_CHECK(v.stride(-1) == 1, "Input tensor must have contiguous last dimension");

    const int total_q = q.size(0);
    const int head_size = k.size(-1);
    const int num_heads_kv = k.size(-2);
    const int num_heads_q = q.size(-1) / head_size;

    TORCH_CHECK(head_size <= 256, "head_size must be <= 256");
    TORCH_CHECK(head_size % 8 == 0, "head_size must be divisible by 8");
    TORCH_CHECK(num_heads_q % num_heads_kv == 0, "num_heads_q must be divisible by num_heads_kv");

    Flash_fwd_params params;

    params.is_bf16 = q.dtype() == torch::kBFloat16;

    // Set the pointers and strides.
    params.q_ptr = q.data_ptr();
    params.k_ptr = k.data_ptr();
    params.v_ptr = v.data_ptr();
    params.o_ptr = out.data_ptr();
    params.atoms = reinterpret_cast<AttentionAtom*>(attention_atoms.data_ptr());

    // All stride are in elements, not bytes.
    params.q_row_stride = q.stride(0);
    params.k_row_stride = k.stride(1);
    params.v_row_stride = v.stride(1);
    params.o_row_stride = out.stride(0);

    // Assume heads are contiguous.
    params.q_head_stride = head_size;
    params.k_head_stride = head_size;
    params.v_head_stride = head_size;
    params.o_head_stride = head_size;

    // Head params
    params.h = num_heads_q;
    params.h_k = num_heads_kv;
    params.h_h_k_ratio = num_heads_q / num_heads_kv;
    params.d = head_size;
    auto round_multiple = [](int x, int m) { return (x + m - 1) / m * m; };
    params.d_rounded = round_multiple(head_size, 32);
    params.num_atoms = attention_atoms.size(0);

    // Set the different scale values.
    params.scale_softmax = softmax_scale;
    params.scale_softmax_log2 = softmax_scale * M_LOG2E;

    params.is_causal = is_causal;

    auto stream = at::cuda::getCurrentCUDAStream().stream();
    run_mha_fwd(params, stream);
}
