#pragma once

#include <assert.h>
#include <cublas_v2.h>
#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <mma.h>
#include <stdio.h>
#include "cublas_wrappers.h"

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
{
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
                                         (transb == CUBLAS_OP_N) ? k : n,
                                         (const void*)beta,
                                         C,
                                         CUDA_R_32F,
                                         m,
                                         CUDA_R_32F,
                                         algo);

    if (status != CUBLAS_STATUS_SUCCESS) {
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
{
    cublasStatus_t status = cublasGemmEx(handle,
                                         transa,
                                         transb,
                                         m,
                                         n,
                                         k,
                                         (const void*)alpha,
                                         (const void*)A,
                                         CUDA_R_16F,
                                         (transa == CUBLAS_OP_N) ? m : k,
                                         (const void*)B,
                                         CUDA_R_16F,
                                         (transb == CUBLAS_OP_N) ? k : n,
                                         (const void*)beta,
                                         (void*)C,
                                         CUDA_R_16F,
                                         m,
                                         CUDA_R_32F,
                                         algo);

    if (status != CUBLAS_STATUS_SUCCESS) {
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
{
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

    if (status != CUBLAS_STATUS_SUCCESS) {
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
{
    cublasStatus_t status = cublasGemmStridedBatchedEx(handle,
                                                       op_A,
                                                       op_B,
                                                       m,
                                                       n,
                                                       k,
                                                       alpha,
                                                       A,
                                                       CUDA_R_16F,
                                                       (op_A == CUBLAS_OP_N) ? m : k,
                                                       stride_A,
                                                       B,
                                                       CUDA_R_16F,
                                                       (op_B == CUBLAS_OP_N) ? k : n,
                                                       stride_B,
                                                       beta,
                                                       C,
                                                       CUDA_R_16F,
                                                       m,
                                                       stride_C,
                                                       batch,
                                                       CUDA_R_32F,
                                                       algo);

    if (status != CUBLAS_STATUS_SUCCESS) {
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
