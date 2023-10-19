// Copyright (c) Microsoft Corporation.
// SPDX-License-Identifier: Apache-2.0

// DeepSpeed Team

#pragma once

#include <assert.h>
#include <cublas_v2.h>
#include <cuda.h>
#ifdef BF16_AVAILABLE
#include <cuda_bf16.h>
#endif
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#ifndef __HIP_PLATFORM_AMD__
#include <mma.h>
#endif
#include <stdio.h>

#ifdef __HIP_PLATFORM_AMD__
int cublas_gemm_ex(rocblas_handle handle,
                   rocblas_operation transa,
                   rocblas_operation transb,
                   int m,
                   int n,
                   int k,
                   const float* alpha,
                   const float* beta,
                   const float* A,
                   const float* B,
                   float* C,
                   rocblas_gemm_algo algo,
                   int b_stride = -1)
#else
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
                   cublasGemmAlgo_t algo,
                   int b_stride = -1)
#endif
{
    const int ldb = (b_stride == -1) ? ((transb == CUBLAS_OP_N) ? k : n) : b_stride;
#ifdef __HIP_PLATFORM_AMD__
    rocblas_status status = rocblas_gemm_ex(handle,
                                            transa,
                                            transb,
                                            m,
                                            n,
                                            k,
                                            (const void*)alpha,
                                            (const void*)A,
                                            rocblas_datatype_f32_r,
                                            (transa == rocblas_operation_none) ? m : k,
                                            (const void*)B,
                                            rocblas_datatype_f32_r,
                                            ldb,
                                            (const void*)beta,
                                            C,
                                            rocblas_datatype_f32_r,
                                            m,
                                            C,
                                            rocblas_datatype_f32_r,
                                            m,
                                            rocblas_datatype_f32_r,
                                            algo,
                                            0,
                                            0);
#else
    cublasStatus_t status = cublasGemmEx(handle,
                                         transa,
                                         transb,
                                         m,
                                         n,
                                         k,
                                         (const void*)alpha,
                                         (const void*)A,
                                         CUDA_R_32F,
                                         (transa == CUBLAS_OP_N) ? m : k,
                                         (const void*)B,
                                         CUDA_R_32F,
                                         ldb,
                                         (const void*)beta,
                                         C,
                                         CUDA_R_32F,
                                         m,
                                         CUDA_R_32F,
                                         algo);
#endif

#ifdef __HIP_PLATFORM_AMD__
    if (status != rocblas_status_success) {
#else
    if (status != CUBLAS_STATUS_SUCCESS) {
#endif
        fprintf(stderr,
                "!!!! kernel execution error. (m: %d, n: %d, k: %d, error: %d) \n",
                m,
                n,
                k,
                (int)status);
        return EXIT_FAILURE;
    }
    return 0;
}

template <typename T>
#ifdef __HIP_PLATFORM_AMD__
int cublas_gemm_ex(rocblas_handle handle,
                   rocblas_operation transa,
                   rocblas_operation transb,
                   int m,
                   int n,
                   int k,
                   const float* alpha,
                   const float* beta,
                   const T* A,
                   const T* B,
                   T* C,
                   rocblas_gemm_algo algo,
                   int b_stride = -1)
#else
int cublas_gemm_ex(cublasHandle_t handle,
                   cublasOperation_t transa,
                   cublasOperation_t transb,
                   int m,
                   int n,
                   int k,
                   const float* alpha,
                   const float* beta,
                   const T* A,
                   const T* B,
                   T* C,
                   cublasGemmAlgo_t algo,
                   int b_stride = -1)
#endif
{
    const int ldb = (b_stride == -1) ? ((transb == CUBLAS_OP_N) ? k : n) : b_stride;
#ifdef __HIP_PLATFORM_AMD__
    constexpr auto rocblas_dtype_16 = std::is_same<T, __half>::value ? rocblas_datatype_f16_r
                                                                     : rocblas_datatype_bf16_r;
    rocblas_status status = rocblas_gemm_ex(handle,
                                            transa,
                                            transb,
                                            m,
                                            n,
                                            k,
                                            (const void*)alpha,
                                            (const void*)A,
                                            rocblas_dtype_16,
                                            (transa == rocblas_operation_none) ? m : k,
                                            (const void*)B,
                                            rocblas_dtype_16,
                                            ldb,
                                            (const void*)beta,
                                            (void*)C,
                                            rocblas_dtype_16,
                                            m,
                                            (void*)C,
                                            rocblas_dtype_16,
                                            m,
                                            rocblas_datatype_f32_r,
                                            algo,
                                            0,
                                            0);
#else
    constexpr auto cublas_dtype_16 = std::is_same<T, __half>::value ? CUDA_R_16F : CUDA_R_16BF;
    cublasStatus_t status = cublasGemmEx(handle,
                                         transa,
                                         transb,
                                         m,
                                         n,
                                         k,
                                         (const void*)alpha,
                                         (const void*)A,
                                         cublas_dtype_16,
                                         (transa == CUBLAS_OP_N) ? m : k,
                                         (const void*)B,
                                         cublas_dtype_16,
                                         ldb,
                                         (const void*)beta,
                                         (void*)C,
                                         cublas_dtype_16,
                                         m,
                                         CUDA_R_32F,
                                         algo);
#endif

#ifdef __HIP_PLATFORM_AMD__
    if (status != rocblas_status_success) {
#else
    if (status != CUBLAS_STATUS_SUCCESS) {
#endif
        fprintf(stderr,
                "!!!! kernel execution error. (m: %d, n: %d, k: %d, error: %d) \n",
                m,
                n,
                k,
                (int)status);
        return EXIT_FAILURE;
    }
    return 0;
}

#ifdef __HIP_PLATFORM_AMD__
int cublas_strided_batched_gemm(rocblas_handle handle,
                                int m,
                                int n,
                                int k,
                                const float* alpha,
                                const float* beta,
                                const float* A,
                                const float* B,
                                float* C,
                                rocblas_operation op_A,
                                rocblas_operation op_B,
                                int stride_A,
                                int stride_B,
                                int stride_C,
                                int batch,
                                rocblas_gemm_algo algo)
#else
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
                                cublasGemmAlgo_t algo)
#endif
{
#ifdef __HIP_PLATFORM_AMD__
    rocblas_status status =
        rocblas_gemm_strided_batched_ex(handle,
                                        op_A,
                                        op_B,
                                        m,
                                        n,
                                        k,
                                        alpha,
                                        A,
                                        rocblas_datatype_f32_r,
                                        (op_A == rocblas_operation_none) ? m : k,
                                        stride_A,
                                        B,
                                        rocblas_datatype_f32_r,
                                        (op_B == rocblas_operation_none) ? k : n,
                                        stride_B,
                                        beta,
                                        C,
                                        rocblas_datatype_f32_r,
                                        m,
                                        stride_C,
                                        C,
                                        rocblas_datatype_f32_r,
                                        m,
                                        stride_C,
                                        batch,
                                        rocblas_datatype_f32_r,
                                        algo,
                                        0,
                                        0);
#else
    cublasStatus_t status = cublasGemmStridedBatchedEx(handle,
                                                       op_A,
                                                       op_B,
                                                       m,
                                                       n,
                                                       k,
                                                       alpha,
                                                       A,
                                                       CUDA_R_32F,
                                                       (op_A == CUBLAS_OP_N) ? m : k,
                                                       stride_A,
                                                       B,
                                                       CUDA_R_32F,
                                                       (op_B == CUBLAS_OP_N) ? k : n,
                                                       stride_B,
                                                       beta,
                                                       C,
                                                       CUDA_R_32F,
                                                       m,
                                                       stride_C,
                                                       batch,
                                                       CUDA_R_32F,
                                                       algo);
#endif

#ifdef __HIP_PLATFORM_AMD__
    if (status != rocblas_status_success) {
#else
    if (status != CUBLAS_STATUS_SUCCESS) {
#endif
        fprintf(stderr,
                "!!!! kernel execution error. (batch: %d, m: %d, n: %d, k: %d, error: %d) \n",
                batch,
                m,
                n,
                k,
                (int)status);
        return EXIT_FAILURE;
    }
    return 0;
}

template <typename T>
#ifdef __HIP_PLATFORM_AMD__
int cublas_strided_batched_gemm(rocblas_handle handle,
                                int m,
                                int n,
                                int k,
                                const float* alpha,
                                const float* beta,
                                const T* A,
                                const T* B,
                                T* C,
                                rocblas_operation op_A,
                                rocblas_operation op_B,
                                int stride_A,
                                int stride_B,
                                int stride_C,
                                int batch,
                                rocblas_gemm_algo algo)
#else
int cublas_strided_batched_gemm(cublasHandle_t handle,
                                int m,
                                int n,
                                int k,
                                const float* alpha,
                                const float* beta,
                                const T* A,
                                const T* B,
                                T* C,
                                cublasOperation_t op_A,
                                cublasOperation_t op_B,
                                int stride_A,
                                int stride_B,
                                int stride_C,
                                int batch,
                                cublasGemmAlgo_t algo)
#endif
{
#ifdef __HIP_PLATFORM_AMD__
    constexpr auto rocblas_dtype_16 = std::is_same<T, __half>::value ? rocblas_datatype_f16_r
                                                                     : rocblas_datatype_bf16_r;
    rocblas_status status =
        rocblas_gemm_strided_batched_ex(handle,
                                        op_A,
                                        op_B,
                                        m,
                                        n,
                                        k,
                                        alpha,
                                        A,
                                        rocblas_dtype_16,
                                        (op_A == rocblas_operation_none) ? m : k,
                                        stride_A,
                                        B,
                                        rocblas_dtype_16,
                                        (op_B == rocblas_operation_none) ? k : n,
                                        stride_B,
                                        beta,
                                        C,
                                        rocblas_dtype_16,
                                        m,
                                        stride_C,
                                        C,
                                        rocblas_dtype_16,
                                        m,
                                        stride_C,
                                        batch,
                                        rocblas_datatype_f32_r,
                                        algo,
                                        0,
                                        0);
#else
    constexpr auto cublas_dtype_16 = std::is_same<T, __half>::value ? CUDA_R_16F : CUDA_R_16BF;
    cublasStatus_t status = cublasGemmStridedBatchedEx(handle,
                                                       op_A,
                                                       op_B,
                                                       m,
                                                       n,
                                                       k,
                                                       alpha,
                                                       A,
                                                       cublas_dtype_16,
                                                       (op_A == CUBLAS_OP_N) ? m : k,
                                                       stride_A,
                                                       B,
                                                       cublas_dtype_16,
                                                       (op_B == CUBLAS_OP_N) ? k : n,
                                                       stride_B,
                                                       beta,
                                                       C,
                                                       cublas_dtype_16,
                                                       m,
                                                       stride_C,
                                                       batch,
                                                       CUDA_R_32F,
                                                       algo);
#endif

#ifdef __HIP_PLATFORM_AMD__
    if (status != rocblas_status_success) {
#else
    if (status != CUBLAS_STATUS_SUCCESS) {
#endif
        fprintf(stderr,
                "!!!! kernel execution error. (m: %d, n: %d, k: %d, error: %d) \n",
                m,
                n,
                k,
                (int)status);
        return EXIT_FAILURE;
    }

    return 0;
}
