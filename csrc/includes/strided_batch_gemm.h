#pragma once

#include <cuda.h>
#include <cuda_fp16.h>
#include <stdio.h>
#include "context.h"

template <typename T>
class StridedBatchGemm {
public:
    struct Config {
        int batch_size;
        int m;
        int n;
        int k;
        float alpha;
        float beta;
        cublasOperation_t op_A;
        cublasOperation_t op_B;
        std::array<int, 3> gemm_algos;

        Config(int batch,
               int mm,
               int nn,
               int kk,
               float param_alpha,
               float param_beta,
               cublasOperation_t opA,
               cublasOperation_t opB,
               const std::array<int, 3>& algos)
            : batch_size(batch),
              m(mm),
              n(nn),
              k(kk),
              alpha(param_alpha),
              beta(param_beta),
              op_A(opA),
              op_B(opB),
              gemm_algos(algos)
        {
        }
        void SetConfig(int mm, int nn, int kk)
        {
            m = mm;
            n = nn;
            k = kk;
        }
    };

    StridedBatchGemm(const Config& config) : _config(config) {}

    virtual ~StridedBatchGemm() {}

    void Forward(int bsz, T* output, const T* _buffer_a, const T* _buffer_b, cublasHandle_t handle)
    {
        int stride_a = _config.m * _config.k;
        int stride_b = _config.n * _config.k;
        int stride_c = _config.m * _config.n;

        cublas_strided_batched_gemm(handle,
                                    _config.m,
                                    _config.n,
                                    _config.k,
                                    &_config.alpha,
                                    &_config.beta,
                                    _buffer_a,
                                    _buffer_b,
                                    output,
                                    _config.op_A,
                                    _config.op_B,
                                    stride_a,
                                    stride_b,
                                    stride_c,
                                    bsz,
                                    cublasGemmAlgo_t(_config.gemm_algos[0]));
    }

    void ForwardPlusSave(T* output, const T* _buffer_a, const T* _buffer_b, cublasHandle_t handle)
    {
        int stride_a = _config.m * _config.k;
        int stride_b = _config.n * _config.k;
        int stride_c = _config.m * _config.n;

        cublas_strided_batched_gemm(handle,
                                    _config.m,
                                    _config.n,
                                    _config.k,
                                    &_config.alpha,
                                    &_config.beta,
                                    _buffer_a,
                                    _buffer_b,
                                    output,
                                    _config.op_A,
                                    _config.op_B,
                                    stride_a,
                                    stride_b,
                                    stride_c,
                                    _config.batch_size,
                                    cublasGemmAlgo_t(_config.gemm_algos[0]));

        k_buf = _buffer_a;
        q_buf = _buffer_b;
    }

    void Backward(int bsz,
                  const T* d_output,
                  const T* _buffer_a,
                  const T* _buffer_b,
                  cublasHandle_t handle,
                  T* inpGradA = nullptr,
                  T* inpGradB = nullptr)
    {
        int mb = (_config.op_A == CUBLAS_OP_T ? _config.k : _config.m);
        int kb = (_config.op_A == CUBLAS_OP_T ? _config.m : _config.k);

        int stride_a = mb * _config.n;
        int stride_b = _config.n * kb;
        int stride_c = _config.m * _config.k;

        // B need to transpose.
        cublasOperation_t op_b = (_config.op_B == CUBLAS_OP_T ? CUBLAS_OP_N : CUBLAS_OP_T);

        // Calculate d_A.
        cublas_strided_batched_gemm(handle,
                                    mb,
                                    kb,
                                    _config.n,
                                    &_config.alpha,
                                    &_config.beta,
                                    (_config.op_A == CUBLAS_OP_T ? _buffer_b : d_output),
                                    (_config.op_A == CUBLAS_OP_T ? d_output : _buffer_b),
                                    inpGradA,
                                    CUBLAS_OP_N,
                                    op_b,
                                    stride_a,
                                    stride_b,
                                    stride_c,
                                    bsz,
                                    cublasGemmAlgo_t(_config.gemm_algos[1]));

        // A need to transpose.
        cublasOperation_t op_a = (_config.op_A == CUBLAS_OP_T ? CUBLAS_OP_N : CUBLAS_OP_T);

        stride_a = _config.m * _config.k;
        stride_b = _config.m * _config.n;
        stride_c = _config.n * _config.k;

        // Calculate d_B.
        cublas_strided_batched_gemm(handle,
                                    _config.k,
                                    _config.n,
                                    _config.m,
                                    &_config.alpha,
                                    &_config.beta,
                                    _buffer_a,
                                    d_output,
                                    inpGradB,
                                    op_a,
                                    CUBLAS_OP_N,
                                    stride_a,
                                    stride_b,
                                    stride_c,
                                    bsz,
                                    cublasGemmAlgo_t(_config.gemm_algos[2]));
    }

    inline int GetN() const { return _config.k; }

    inline const T* GetBufferA() const { return k_buf; }

    inline const T* GetBufferB() const { return q_buf; }

    inline void SetConfig(int m, int n, int k) { _config.SetConfig(m, n, k); }

private:
    Config _config;
    const T* q_buf;
    const T* k_buf;
};
