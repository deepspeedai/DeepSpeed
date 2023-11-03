// Copyright (c) Microsoft Corporation.
// SPDX-License-Identifier: Apache-2.0

// DeepSpeed Team

#pragma once

#include <c10/cuda/CUDAStream.h>
#include <torch/extension.h>
#include <cstdio>
#include "blas_utils.h"

#define DISPATCH_BLAS_MATMUL(T_TYPE, C_TYPE)                \
    if (output.options().dtype() == torch::T_TYPE) {        \
        blas_gemm_ex(output.data_ptr(),                     \
                     (const void*)weights.data_ptr(),       \
                     (const void*)hidden_states.data_ptr(), \
                     m,                                     \
                     n,                                     \
                     k,                                     \
                     lda,                                   \
                     ldb,                                   \
                     ldc,                                   \
                     trans_a,                               \
                     trans_b,                               \
                     &alpha,                                \
                     &beta,                                 \
                     C_TYPE);                               \
    }

void blas_linear(at::Tensor& output, at::Tensor& hidden_states, at::Tensor& weights)
{
    /*
    Expected shape: output([total_tokens_across_dims], out_neurons)
                    hidden_states([total_tokens_across_dims], in_neurons)
                    weights(out_neurons, in_neurons)

    We are going to assume contiguous for the above shapes.

    The shapes are going to get messed with a little internally to handle column-major
    GEMMs.
    */

    // Number of tokens is N (since the GEMM output is column-major but our Tensor
    // is row-major, we need to transpose the shapes)
    const int n = output.numel() / output.size(-1);
    const int k = weights.size(1);
    const int m = weights.size(0);

    // A strides
    const bool trans_a = weights.stride(1) == 1;
    const int lda = (trans_a) ? weights.stride(0) : weights.stride(1);

    // B strides
    const bool trans_b = hidden_states.stride(-1) != 1;
    const int ldb = (trans_b) ? hidden_states.stride(-1) : hidden_states.stride(-2);

    // C strides
    const int ldc = output.stride(-2);

    const float alpha = 1.0f;
    const float beta = 0.0f;

    TORCH_CHECK(output.scalar_type() == hidden_states.scalar_type(),
                "Output and hidden states must have the same scalar type");
    TORCH_CHECK(output.scalar_type() == weights.scalar_type(),
                "Output and weights must have the same scalar type");

    // Dispatch the datatypes
    DISPATCH_BLAS_MATMUL(kFloat, BlasType::FP32);
    DISPATCH_BLAS_MATMUL(kHalf, BlasType::FP16);
#ifdef BF16_AVAILABLE
    DISPATCH_BLAS_MATMUL(kBFloat16, BlasType::BF16);
#endif
}

#define DISPATCH_4D_BLAS(T_TYPE, C_TYPE)                     \
    if (C.options().dtype() == torch::T_TYPE) {              \
        blas_strided_batched_gemm(C.data_ptr(),              \
                                  (const void*)A.data_ptr(), \
                                  (const void*)B.data_ptr(), \
                                  m,                         \
                                  n,                         \
                                  k,                         \
                                  lda,                       \
                                  ldb,                       \
                                  ldc,                       \
                                  trans_a,                   \
                                  trans_b,                   \
                                  &alpha,                    \
                                  &beta,                     \
                                  stride_a,                  \
                                  stride_b,                  \
                                  stride_c,                  \
                                  batch,                     \
                                  C_TYPE);                   \
    }

void blas_4d_matmul(at::Tensor& C, at::Tensor& B, at::Tensor& A)
{
    /*
    C shape: (batch_size, N, M)
    A shape: (batch_size, N, K)
    B shape: (batch_size, K, M)
    */

    const int n = C.size(-2);
    const int k = C.size(-1);
    const int m = B.size(-1);

    // A strides
    const bool trans_a = A.stride(-1) == 1;
    const int lda = (trans_a) ? A.stride(-2) : A.stride(-1);
    const int stride_a = A.stride(-3);

    // B strides
    const bool trans_b = B.stride(-1) != 1;
    const int ldb = (trans_b) ? B.stride(-1) : B.stride(-2);
    const int stride_b = B.stride(-3);

    // C strides
    const int ldc = C.stride(-2);
    const int stride_c = C.stride(-3);

    const float alpha = 1.0f;
    const float beta = 0.0f;

    const int batch = C.numel() / (n * m);

    // Dispatch the datatypes
    DISPATCH_4D_BLAS(kFloat, BlasType::FP32);
    DISPATCH_4D_BLAS(kHalf, BlasType::FP16);
#ifdef BF16_AVAILABLE
    DISPATCH_4D_BLAS(kBFloat16, BlasType::BF16);
#endif
}

void create_handle() { BlasContext::getInstance().get_handle(); }
