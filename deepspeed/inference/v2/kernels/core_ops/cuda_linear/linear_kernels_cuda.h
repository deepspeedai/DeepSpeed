// Copyright (c) Microsoft Corporation.
// SPDX-License-Identifier: Apache-2.0

// DeepSpeed Team

// This is a copy of FP6-LLM kernel code: https://arxiv.org/abs/2401.14112

#ifndef DEEPSPEED_CUDA_LINEAR_FP6_LINEAR_CUH
#define DEEPSPEED_CUDA_LINEAR_FP6_LINEAR_CUH

#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <torch/extension.h>

/*
 * Computes FP6-FP16 GEMM (C++ interface).
 */
cudaError_t fp6_linear_kernel(cudaStream_t stream,
                              const uint4* Weight1,
                              const uint4* Weight2,
                              const half* Scales,
                              const half* B,
                              half* C,
                              const size_t M_Global,
                              const size_t N_Global,
                              const size_t K_Global,
                              float* Reduction_Workspace,  // Reduction_Workspace_Size = Split_K *
                                                           // M_Global * N_Global * sizeof(fp32)
                              int Split_K);

/*
 * Computes FP6-FP16 GEMM (PyTorch interface).
 */
torch::Tensor fp6_linear_forward_cuda(torch::Tensor _in_feats,
                                      torch::Tensor _weights,
                                      torch::Tensor _scales,
                                      int splitK = 1);

/*
 * In-place weight prepacking (C++ interface).
 */
void weight_matrix_prepacking(int* FP6Weights, size_t M, size_t K);

/*
 * Weight prepacking (Pytorch interface).
 */
torch::Tensor weight_matrix_prepacking_cpu(torch::Tensor fp6_tensor, size_t M, size_t K);

#endif
