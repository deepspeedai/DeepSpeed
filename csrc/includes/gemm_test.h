// Copyright (c) Microsoft Corporation.
// SPDX-License-Identifier: Apache-2.0

// DeepSpeed Team

#pragma once

#include <cuda_fp16.h>
#ifndef __HIP_PLATFORM_AMD__
#include <cuda_profiler_api.h>
#endif
#ifdef __HIP_PLATFORM_AMD__
#include <rocblas/rocblas.h>
#endif
#include <array>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <limits>
#include <memory>
#include "StopWatch.h"
#include "cublas_wrappers.h"

template <typename T>
void check(T result, char const* const func, const char* const file, int const line)
{
    if (result) {
        std::cout << (std::string("CUDA runtime error: ") + +file + ":" + std::to_string(line) +
                      " \n");
    }
}

#define check_cuda_error(val) check((val), #val, __FILE__, __LINE__)

template <typename T>
class GemmTest {
public:
    GemmTest(int m, int n, int k, cublasOperation_t ta, cublasOperation_t tb, cublasHandle_t h)
        : M(m), N(n), K(k), transa(ta), transb(tb), handle(h)
    {
        check_cuda_error(cudaMalloc((void**)&A, sizeof(T) * M * K));
        check_cuda_error(cudaMalloc((void**)&B, sizeof(T) * K * N));
        check_cuda_error(cudaMalloc((void**)&C, sizeof(T) * M * N));
    }

    ~GemmTest()
    {
        check_cuda_error(cudaFree(A));
        check_cuda_error(cudaFree(B));
        check_cuda_error(cudaFree(C));
    }

    std::array<int, 3> TestAlgo(int loops)
    {
        float alpha = (T)1.0f;
        float beta = (T)0.0f;

        int algo_fw = Run(loops, [=](int algo) {
            cublas_gemm_ex(handle,
                           CUBLAS_OP_T,
                           CUBLAS_OP_N,
                           N,
                           M,
                           K,
                           &alpha,
                           &beta,
                           B,
                           A,
                           C,
#ifdef __HIP_PLATFORM_AMD__
                           static_cast<rocblas_gemm_algo>(algo));
#else
                           static_cast<cublasGemmAlgo_t>(algo));
#endif
        });

        int algo_bw1 = Run(loops, [=](int algo) {
            cublas_gemm_ex(handle,
                           CUBLAS_OP_N,
                           CUBLAS_OP_T,
                           K,
                           N,
                           M,
                           &alpha,
                           &beta,
                           A,
                           C,
                           B,
#ifdef __HIP_PLATFORM_AMD__
                           static_cast<rocblas_gemm_algo>(algo));
#else
                           static_cast<cublasGemmAlgo_t>(algo));
#endif
        });

        int algo_bw2 = Run(loops, [=](int algo) {
            cublas_gemm_ex(handle,
                           CUBLAS_OP_N,
                           CUBLAS_OP_N,
                           K,
                           M,
                           N,
                           &alpha,
                           &beta,
                           B,
                           C,
                           A,
#ifdef __HIP_PLATFORM_AMD__
                           static_cast<rocblas_gemm_algo>(algo));
#else
                           static_cast<cublasGemmAlgo_t>(algo));
#endif
        });

        return std::array<int, 3>({algo_fw, algo_bw1, algo_bw2});
    }

    template <typename Func>
    int Run(int loops, Func f)
    {
        float fast_latency = (std::numeric_limits<float>::max)();
        int fast_algo = 0;

#ifdef __HIP_PLATFORM_AMD__
        for (int algo = (int)rocblas_gemm_algo_standard; algo <= (int)rocblas_gemm_algo_standard;
#else
        for (int algo = (int)CUBLAS_GEMM_DEFAULT_TENSOR_OP;
             algo <= (int)CUBLAS_GEMM_ALGO15_TENSOR_OP;
#endif
             algo++) {
            int warm_up = 5;
            for (int i = 0; i < warm_up; ++i) f(algo);

            cudaDeviceSynchronize();
            Stopwatch timer;
            timer.Restart();

            for (int i = 0; i < loops; ++i) f(algo);

            cudaDeviceSynchronize();
            timer.Stop();

            float avg_latency = (float)timer.GetTimeInSeconds() * 1000 / loops;

            printf("algo-%d: %.3fms\n", algo, avg_latency);

            if (avg_latency < fast_latency) {
                fast_latency = avg_latency;
                fast_algo = algo;
            }
        }

        printf("fast_algo %d: %.3f ms\n", fast_algo, fast_latency);

        return fast_algo;
    }

private:
    int M, N, K;
    cublasHandle_t handle;
    cublasOperation_t transa, transb;
    T *A, *B, *C;
};

template <typename T>
class StridedGemmTest {
public:
    StridedGemmTest(int b,
                    int m,
                    int n,
                    int k,
                    cublasOperation_t ta,
                    cublasOperation_t tb,
                    cublasHandle_t h)
        : bsz(b), M(m), N(n), K(k), transa(ta), transb(tb), handle(h)
    {
        check_cuda_error(cudaMalloc((void**)&A, sizeof(T) * M * K * bsz));
        check_cuda_error(cudaMalloc((void**)&B, sizeof(T) * K * N * bsz));
        check_cuda_error(cudaMalloc((void**)&C, sizeof(T) * M * N * bsz));
    }

    ~StridedGemmTest()
    {
        check_cuda_error(cudaFree(A));
        check_cuda_error(cudaFree(B));
        check_cuda_error(cudaFree(C));
    }

    std::array<int, 3> TestAlgo(int loops)
    {
        float alpha = (T)1.0f;
        float beta = (T)0.0f;

        int algo_fw = Run(loops, [=](int algo) {
            int stride_a = M * K;
            int stride_b = N * K;
            int stride_c = M * N;

            cublas_strided_batched_gemm(handle,
                                        M,
                                        N,
                                        K,
                                        &alpha,
                                        &beta,
                                        A,
                                        B,
                                        C,
                                        transa,
                                        transb,
                                        stride_a,
                                        stride_b,
                                        stride_c,
                                        bsz,
#ifdef __HIP_PLATFORM_AMD__
                                        static_cast<rocblas_gemm_algo>(algo));
#else
                                        static_cast<cublasGemmAlgo_t>(algo));
#endif
        });

        int algo_bw1 = Run(loops, [=](int algo) {
            int mb = (transa == CUBLAS_OP_T ? K : M);
            int kb = (transa == CUBLAS_OP_T ? M : K);

            int stride_a = mb * N;
            int stride_b = N * kb;
            int stride_c = M * K;

            // B need to transpose.
            cublasOperation_t op_b = (transb == CUBLAS_OP_T ? CUBLAS_OP_N : CUBLAS_OP_T);

            // Calculate d_A.
            cublas_strided_batched_gemm(handle,
                                        mb,
                                        kb,
                                        N,
                                        &alpha,
                                        &beta,
                                        (transa == CUBLAS_OP_T ? B : C),
                                        (transa == CUBLAS_OP_T ? C : B),
                                        A,
                                        CUBLAS_OP_N,
                                        op_b,
                                        stride_a,
                                        stride_b,
                                        stride_c,
                                        bsz,
#ifdef __HIP_PLATFORM_AMD__
                                        static_cast<rocblas_gemm_algo>(algo));
#else
                                        static_cast<cublasGemmAlgo_t>(algo));
#endif
        });

        int algo_bw2 = Run(loops, [=](int algo) {
            // A need to transpose.
            cublasOperation_t op_a = (transa == CUBLAS_OP_T ? CUBLAS_OP_N : CUBLAS_OP_T);

            int stride_a = M * K;
            int stride_b = M * N;
            int stride_c = N * K;

            // Calculate d_B.
            cublas_strided_batched_gemm(handle,
                                        K,
                                        N,
                                        M,
                                        &alpha,
                                        &beta,
                                        A,
                                        C,
                                        B,
                                        op_a,
                                        CUBLAS_OP_N,
                                        stride_a,
                                        stride_b,
                                        stride_c,
                                        bsz,
#ifdef __HIP_PLATFORM_AMD__
                                        static_cast<rocblas_gemm_algo>(algo));
#else
                                        static_cast<cublasGemmAlgo_t>(algo));
#endif
        });

        return std::array<int, 3>({algo_fw, algo_bw1, algo_bw2});
    }

    template <typename Func>
    int Run(int loops, Func f)
    {
        float fast_latency = (std::numeric_limits<float>::max)();
        int fast_algo = 0;

#ifdef __HIP_PLATFORM_AMD__
        for (int algo = (int)rocblas_gemm_algo_standard; algo <= (int)rocblas_gemm_algo_standard;
#else
        for (int algo = (int)CUBLAS_GEMM_DEFAULT_TENSOR_OP;
             algo <= (int)CUBLAS_GEMM_ALGO15_TENSOR_OP;
#endif
             algo++) {
            int warm_up = 5;
            for (int i = 0; i < warm_up; ++i) f(algo);

            cudaDeviceSynchronize();
            Stopwatch timer;
            timer.Restart();

            for (int i = 0; i < loops; ++i) f(algo);

            cudaDeviceSynchronize();
            timer.Stop();

            float avg_latency = (float)timer.GetTimeInSeconds() * 1000 / loops;

            printf("algo-%d: %.3fms\n", algo, avg_latency);

            if (avg_latency < fast_latency) {
                fast_latency = avg_latency;
                fast_algo = algo;
            }
        }

        printf("fast_algo %d: %.3f ms\n", fast_algo, fast_latency);

        return fast_algo;
    }

private:
    int bsz, M, N, K;
    cublasHandle_t handle;
    cublasOperation_t transa, transb;
    T *A, *B, *C;
};
