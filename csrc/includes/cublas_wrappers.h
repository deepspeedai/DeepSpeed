// Copyright (c) Microsoft Corporation.
// SPDX-License-Identifier: Apache-2.0

// DeepSpeed Team

#pragma once

#include <assert.h>
#include <cublas_v2.h>
#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#ifndef __HIP_PLATFORM_AMD__
#include <mma.h>
#endif
#ifdef __HIP_PLATFORM_AMD__
#include <rocblas/rocblas.h>
#endif
#include <stdio.h>
#include <torch/version.h>

int cublas_gemm_ex(cublasHandle_t handle,
                   cublasOperation_t transa,
                   cublasOperation_t transb,
                   int m,
                   int n,
                   int k,
                   const float* alpha,
                   const float* beta,
                   const float* A,
                   const float* B,
                   float* C,
// TODO HIP: Remove backward compatibility for torch<=2.0 in future
#if defined(__HIP_PLATFORM_AMD__) && \
    ((TORCH_VERSION_MAJOR < 2) || (TORCH_VERSION_MAJOR == 2 && TORCH_VERSION_MINOR == 0))
                   rocblas_gemm_algo algo = rocblas_gemm_algo_standard);
#else
                   cublasGemmAlgo_t algo = CUBLAS_GEMM_DEFAULT);
#endif

int cublas_gemm_ex(cublasHandle_t handle,
                   cublasOperation_t transa,
                   cublasOperation_t transb,
                   int m,
                   int n,
                   int k,
                   const float* alpha,
                   const float* beta,
                   const __half* A,
                   const __half* B,
                   __half* C,
#if defined(__HIP_PLATFORM_AMD__) && \
    ((TORCH_VERSION_MAJOR < 2) || (TORCH_VERSION_MAJOR == 2 && TORCH_VERSION_MINOR == 0))
                   rocblas_gemm_algo algo = rocblas_gemm_algo_standard);
#else
                   cublasGemmAlgo_t algo = CUBLAS_GEMM_DEFAULT_TENSOR_OP);
#endif

int cublas_strided_batched_gemm(cublasHandle_t handle,
                                int m,
                                int n,
                                int k,
                                const float* alpha,
                                const float* beta,
                                const float* A,
                                const float* B,
                                float* C,
                                cublasOperation_t op_A,
                                cublasOperation_t op_B,
                                int stride_A,
                                int stride_B,
                                int stride_C,
                                int batch,
#if defined(__HIP_PLATFORM_AMD__) && \
    ((TORCH_VERSION_MAJOR < 2) || (TORCH_VERSION_MAJOR == 2 && TORCH_VERSION_MINOR == 0))
                                rocblas_gemm_algo algo = rocblas_gemm_algo_standard);
#else
                                cublasGemmAlgo_t algo = CUBLAS_GEMM_DEFAULT);
#endif

int cublas_strided_batched_gemm(cublasHandle_t handle,
                                int m,
                                int n,
                                int k,
                                const float* alpha,
                                const float* beta,
                                const __half* A,
                                const __half* B,
                                __half* C,
                                cublasOperation_t op_A,
                                cublasOperation_t op_B,
                                int stride_A,
                                int stride_B,
                                int stride_C,
                                int batch,
#if defined(__HIP_PLATFORM_AMD__) && \
    ((TORCH_VERSION_MAJOR < 2) || (TORCH_VERSION_MAJOR == 2 && TORCH_VERSION_MINOR == 0))
                                rocblas_gemm_algo algo = rocblas_gemm_algo_standard);
#else
                                cublasGemmAlgo_t algo = CUBLAS_GEMM_DEFAULT_TENSOR_OP);
#endif
