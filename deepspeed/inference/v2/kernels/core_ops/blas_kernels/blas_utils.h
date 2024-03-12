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
#include <iostream>
#include <stdexcept>

class BlasContext {
    /*
    Slim wrapper for managing the lifetime of the platform's BLAS handle. This should
    be hipified for ROCm.
    */
public:
    BlasContext()
    {
        if (cublasCreate(&_handle) != CUBLAS_STATUS_SUCCESS) {
            auto message = std::string("Fail to create cublas handle.");
            std::cerr << message << std::endl;
            throw std::runtime_error(message);
        }
#ifndef __HIP_PLATFORM_AMD__
        cublasSetMathMode(_handle, CUBLAS_TENSOR_OP_MATH);
#endif
    }

    virtual ~BlasContext() { cublasDestroy(_handle); }

    static BlasContext& getInstance()
    {
        // Should always access the singleton through this function.
        static BlasContext _instance;
        return _instance;
    }

    cublasHandle_t get_handle() const { return _handle; }

private:
    cublasHandle_t _handle;
};

enum class BlasType { FP32, FP16, BF16 };

#ifdef __HIP_PLATFORM_AMD__
rocblas_operation get_trans_op(bool do_trans)
{
    return (do_trans) ? rocblas_operation_transpose : rocblas_operation_none;
}

rocblas_datatype get_datatype(BlasType type)
{
    switch (type) {
        case BlasType::FP32: return rocblas_datatype_f32_r;
        case BlasType::FP16: return rocblas_datatype_f16_r;
        case BlasType::BF16: return rocblas_datatype_bf16_r;
        default: throw std::runtime_error("Unsupported BlasType");
    }
}
#else
cublasOperation_t get_trans_op(bool do_trans) { return (do_trans) ? CUBLAS_OP_T : CUBLAS_OP_N; }

cublasDataType_t get_datatype(BlasType type)
{
    switch (type) {
        case BlasType::FP32: return CUDA_R_32F;
        case BlasType::FP16: return CUDA_R_16F;
        case BlasType::BF16: return CUDA_R_16BF;
        default: throw std::runtime_error("Unsupported BlasType");
    }
}
#endif

int blas_gemm_ex(void* C,
                 const void* A,
                 const void* B,
                 int m,
                 int n,
                 int k,
                 int lda,
                 int ldb,
                 int ldc,
                 bool transa,
                 bool transb,
                 const float* alpha,
                 const float* beta,
                 BlasType type)
{
#ifdef __HIP_PLATFORM_AMD__
    rocblas_operation_t transa_op = get_trans_op(transa);
    rocblas_operation_t transb_op = get_trans_op(transb);

    rocblas_datatype_t abc_type = get_datatype(type);

    rocblas_status status = rocblas_gemm_ex(BlasContext::getInstance().get_handle(),
                                            transa_op,
                                            transb_op,
                                            m,
                                            n,
                                            k,
                                            (const void*)alpha,
                                            A,
                                            abc_type,
                                            lda,
                                            B,
                                            abc_type,
                                            ldb,
                                            (const void*)beta,
                                            C,
                                            abc_type,
                                            ldc,
                                            C,
                                            abc_type,
                                            ldc,
                                            rocblas_datatype_f32_r,
                                            rocblas_gemm_algo_standard,
                                            0,
                                            0);
#else
    cublasOperation_t transa_op = get_trans_op(transa);
    cublasOperation_t transb_op = get_trans_op(transb);

    cublasDataType_t abc_type = get_datatype(type);
    cublasStatus_t status = cublasGemmEx(BlasContext::getInstance().get_handle(),
                                         transa_op,
                                         transb_op,
                                         m,
                                         n,
                                         k,
                                         (const void*)alpha,
                                         A,
                                         abc_type,
                                         lda,
                                         B,
                                         abc_type,
                                         ldb,
                                         (const void*)beta,
                                         C,
                                         abc_type,
                                         ldc,
                                         CUDA_R_32F,
                                         CUBLAS_GEMM_DEFAULT_TENSOR_OP);
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

int blas_strided_batched_gemm(void* C,
                              const void* A,
                              const void* B,
                              int m,
                              int n,
                              int k,
                              int lda,
                              int ldb,
                              int ldc,
                              bool transa,
                              bool transb,
                              const float* alpha,
                              const float* beta,
                              int stride_A,
                              int stride_B,
                              int stride_C,
                              int batch,
                              BlasType type)
{
#ifdef __HIP_PLATFORM_AMD__
    rocblas_operation_t transa_op = get_trans_op(transa);
    rocblas_operation_t transb_op = get_trans_op(transb);

    rocblas_datatype_t abc_type = get_datatype(type);

    rocblas_status status =
        rocblas_gemm_strided_batched_ex(BlasContext::getInstance()::get_handle(),
                                        transa_op,
                                        transb_op,
                                        m,
                                        n,
                                        k,
                                        (const void*)alpha,
                                        A,
                                        abc_type,
                                        lda,
                                        stride_A,
                                        B,
                                        abc_type,
                                        ldb,
                                        stride_B,
                                        (const void*)beta,
                                        C,
                                        abc_type,
                                        ldc,
                                        stride_C,
                                        C,
                                        abc_type,
                                        ldc,
                                        stride_C,
                                        batch,
                                        rocblas_datatype_f32_r,
                                        rocblas_gemm_algo_standard,
                                        0,
                                        0);
#else
    cublasOperation_t transa_op = get_trans_op(transa);
    cublasOperation_t transb_op = get_trans_op(transb);

    cublasDataType_t abc_type = get_datatype(type);

    cublasStatus_t status = cublasGemmStridedBatchedEx(BlasContext::getInstance().get_handle(),
                                                       transa_op,
                                                       transb_op,
                                                       m,
                                                       n,
                                                       k,
                                                       (const void*)alpha,
                                                       A,
                                                       abc_type,
                                                       lda,
                                                       stride_A,
                                                       B,
                                                       abc_type,
                                                       ldb,
                                                       stride_B,
                                                       (const void*)beta,
                                                       C,
                                                       abc_type,
                                                       ldc,
                                                       stride_C,
                                                       batch,
                                                       CUDA_R_32F,
                                                       CUBLAS_GEMM_DEFAULT_TENSOR_OP);
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
