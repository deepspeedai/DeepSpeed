// Copyright (c) Microsoft Corporation.
// SPDX-License-Identifier: Apache-2.0

// DeepSpeed Team

#include "cublas_wrappers.h"

// TODO HIP: Remove backward compatibility for torch<=2.0 in future
#if defined(__HIP_PLATFORM_AMD__) && \
    ((TORCH_VERSION_MAJOR < 2) || (TORCH_VERSION_MAJOR == 2 && TORCH_VERSION_MINOR == 0))
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
                   rocblas_gemm_algo algo)
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
                   cublasGemmAlgo_t algo)
#endif
{
#if defined(__HIP_PLATFORM_AMD__) && \
    ((TORCH_VERSION_MAJOR < 2) || (TORCH_VERSION_MAJOR == 2 && TORCH_VERSION_MINOR == 0))
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
                                            (transb == rocblas_operation_none) ? k : n,
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
#ifdef __HIP_PLATFORM_AMD__
                                         HIPBLAS_R_32F,
#else
                                         CUDA_R_32F,
#endif
                                         (transa == CUBLAS_OP_N) ? m : k,
                                         (const void*)B,
#ifdef __HIP_PLATFORM_AMD__
                                         HIPBLAS_R_32F,
#else
                                         CUDA_R_32F,
#endif
                                         (transb == CUBLAS_OP_N) ? k : n,
                                         (const void*)beta,
                                         C,
#ifdef __HIP_PLATFORM_AMD__
                                         HIPBLAS_R_32F,
#else
                                         CUDA_R_32F,
#endif
                                         m,
#if defined(__HIP_PLATFORM_AMD__) && defined(HIPBLAS_V2)
                                         HIPBLAS_COMPUTE_32F,
#elif defined(__HIP_PLATFORM_AMD__)
                                         HIPBLAS_R_32F,
#else
                                         CUDA_R_32F,
#endif
                                         algo);
#endif

#if defined(__HIP_PLATFORM_AMD__) && \
    ((TORCH_VERSION_MAJOR < 2) || (TORCH_VERSION_MAJOR == 2 && TORCH_VERSION_MINOR == 0))
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

#if defined(__HIP_PLATFORM_AMD__) && \
    ((TORCH_VERSION_MAJOR < 2) || (TORCH_VERSION_MAJOR == 2 && TORCH_VERSION_MINOR == 0))
int cublas_gemm_ex(rocblas_handle handle,
                   rocblas_operation transa,
                   rocblas_operation transb,
                   int m,
                   int n,
                   int k,
                   const float* alpha,
                   const float* beta,
                   const __half* A,
                   const __half* B,
                   __half* C,
                   rocblas_gemm_algo algo)
#else
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
                   cublasGemmAlgo_t algo)
#endif
{
#if defined(__HIP_PLATFORM_AMD__) && \
    ((TORCH_VERSION_MAJOR < 2) || (TORCH_VERSION_MAJOR == 2 && TORCH_VERSION_MINOR == 0))
    rocblas_status status = rocblas_gemm_ex(handle,
                                            transa,
                                            transb,
                                            m,
                                            n,
                                            k,
                                            (const void*)alpha,
                                            (const void*)A,
                                            rocblas_datatype_f16_r,
                                            (transa == rocblas_operation_none) ? m : k,
                                            (const void*)B,
                                            rocblas_datatype_f16_r,
                                            (transb == rocblas_operation_none) ? k : n,
                                            (const void*)beta,
                                            (void*)C,
                                            rocblas_datatype_f16_r,
                                            m,
                                            (void*)C,
                                            rocblas_datatype_f16_r,
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
#ifdef __HIP_PLATFORM_AMD__
                                         HIPBLAS_R_16F,
#else
                                         CUDA_R_16F,
#endif
                                         (transa == CUBLAS_OP_N) ? m : k,
                                         (const void*)B,
#ifdef __HIP_PLATFORM_AMD__
                                         HIPBLAS_R_16F,
#else
                                         CUDA_R_16F,
#endif
                                         (transb == CUBLAS_OP_N) ? k : n,
                                         (const void*)beta,
                                         (void*)C,
#ifdef __HIP_PLATFORM_AMD__
                                         HIPBLAS_R_16F,
#else
                                         CUDA_R_16F,
#endif
                                         m,
#if defined(__HIP_PLATFORM_AMD__) && defined(HIPBLAS_V2)
                                         HIPBLAS_COMPUTE_32F,
#elif defined(__HIP_PLATFORM_AMD__)
                                         HIPBLAS_R_32F,
#else
                                         CUDA_R_32F,
#endif
                                         algo);
#endif

#if defined(__HIP_PLATFORM_AMD__) && \
    ((TORCH_VERSION_MAJOR < 2) || (TORCH_VERSION_MAJOR == 2 && TORCH_VERSION_MINOR == 0))
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

#if defined(__HIP_PLATFORM_AMD__) && \
    ((TORCH_VERSION_MAJOR < 2) || (TORCH_VERSION_MAJOR == 2 && TORCH_VERSION_MINOR == 0))
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
#if defined(__HIP_PLATFORM_AMD__) && \
    ((TORCH_VERSION_MAJOR < 2) || (TORCH_VERSION_MAJOR == 2 && TORCH_VERSION_MINOR == 0))
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
#ifdef __HIP_PLATFORM_AMD__
                                                       HIPBLAS_R_32F,
#else
                                                       CUDA_R_32F,
#endif
                                                       (op_A == CUBLAS_OP_N) ? m : k,
                                                       stride_A,
                                                       B,
#ifdef __HIP_PLATFORM_AMD__
                                                       HIPBLAS_R_32F,
#else
                                                       CUDA_R_32F,
#endif
                                                       (op_B == CUBLAS_OP_N) ? k : n,
                                                       stride_B,
                                                       beta,
                                                       C,
#ifdef __HIP_PLATFORM_AMD__
                                                       HIPBLAS_R_32F,
#else
                                                       CUDA_R_32F,
#endif
                                                       m,
                                                       stride_C,
                                                       batch,
#if defined(__HIP_PLATFORM_AMD__) && defined(HIPBLAS_V2)
                                                       HIPBLAS_COMPUTE_32F,
#elif defined(__HIP_PLATFORM_AMD__)
                                                       HIPBLAS_R_32F,
#else
                                                       CUDA_R_32F,
#endif
                                                       algo);
#endif

#if defined(__HIP_PLATFORM_AMD__) && \
    ((TORCH_VERSION_MAJOR < 2) || (TORCH_VERSION_MAJOR == 2 && TORCH_VERSION_MINOR == 0))
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

#if defined(__HIP_PLATFORM_AMD__) && \
    ((TORCH_VERSION_MAJOR < 2) || (TORCH_VERSION_MAJOR == 2 && TORCH_VERSION_MINOR == 0))
int cublas_strided_batched_gemm(rocblas_handle handle,
                                int m,
                                int n,
                                int k,
                                const float* alpha,
                                const float* beta,
                                const __half* A,
                                const __half* B,
                                __half* C,
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
                                const __half* A,
                                const __half* B,
                                __half* C,
                                cublasOperation_t op_A,
                                cublasOperation_t op_B,
                                int stride_A,
                                int stride_B,
                                int stride_C,
                                int batch,
                                cublasGemmAlgo_t algo)
#endif
{
#if defined(__HIP_PLATFORM_AMD__) && \
    ((TORCH_VERSION_MAJOR < 2) || (TORCH_VERSION_MAJOR == 2 && TORCH_VERSION_MINOR == 0))
    rocblas_status status =
        rocblas_gemm_strided_batched_ex(handle,
                                        op_A,
                                        op_B,
                                        m,
                                        n,
                                        k,
                                        alpha,
                                        A,
                                        rocblas_datatype_f16_r,
                                        (op_A == rocblas_operation_none) ? m : k,
                                        stride_A,
                                        B,
                                        rocblas_datatype_f16_r,
                                        (op_B == rocblas_operation_none) ? k : n,
                                        stride_B,
                                        beta,
                                        C,
                                        rocblas_datatype_f16_r,
                                        m,
                                        stride_C,
                                        C,
                                        rocblas_datatype_f16_r,
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
#ifdef __HIP_PLATFORM_AMD__
                                                       HIPBLAS_R_16F,
#else
                                                       CUDA_R_16F,
#endif
                                                       (op_A == CUBLAS_OP_N) ? m : k,
                                                       stride_A,
                                                       B,
#ifdef __HIP_PLATFORM_AMD__
                                                       HIPBLAS_R_16F,
#else
                                                       CUDA_R_16F,
#endif
                                                       (op_B == CUBLAS_OP_N) ? k : n,
                                                       stride_B,
                                                       beta,
                                                       C,
#ifdef __HIP_PLATFORM_AMD__
                                                       HIPBLAS_R_16F,
#else
                                                       CUDA_R_16F,
#endif
                                                       m,
                                                       stride_C,
                                                       batch,
#if defined(__HIP_PLATFORM_AMD__) && defined(HIPBLAS_V2)
                                                       HIPBLAS_COMPUTE_32F,
#elif defined(__HIP_PLATFORM_AMD__)
                                                       HIPBLAS_R_32F,
#else
                                                       CUDA_R_32F,
#endif
                                                       algo);
#endif

#if defined(__HIP_PLATFORM_AMD__) && \
    ((TORCH_VERSION_MAJOR < 2) || (TORCH_VERSION_MAJOR == 2 && TORCH_VERSION_MINOR == 0))
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
