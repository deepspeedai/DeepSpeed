#pragma once

#if (__x86_64__ || __i386__)
#include <cpuid.h>
#include <x86intrin.h>
#endif

#include <cuda_fp16.h>
#include <cuda_runtime_api.h>
#include <stdio.h>
#include <cassert>
#include "context.h"
#include "cublas_v2.h"
#include "cuda.h"
#include "curand.h"

#define CUDA_CHECK(callstr)                                                                    \
    {                                                                                          \
        cudaError_t error_code = callstr;                                                      \
        if (error_code != cudaSuccess) {                                                       \
            std::cerr << "CUDA error " << error_code << " at " << __FILE__ << ":" << __LINE__; \
            assert(0);                                                                         \
        }                                                                                      \
    }

#define TILE (128 * 1024 * 1024)

#if defined(__AVX512__)
#define SIMD_STORE(a, d) _mm512_storeu_ps(a, d)
#define SIMD_LOAD(x) _mm512_loadu_ps(x)
#define SIMD_SET(x) _mm512_set1_ps(x)
#define SIMD_MUL(x, y) _mm512_mul_ps(x, y)
#define SIMD_FMA(x, y, c) _mm512_fmadd_ps(x, y, c)
#define SIMD_SQRT(x) _mm512_sqrt_ps(x)
#define SIMD_DIV(x, y) _mm512_div_ps(x, y)
#define SIMD_WIDTH 16
#else
#if defined(__AVX256__)
#define SIMD_STORE(a, d) _mm256_storeu_ps(a, d)
#define SIMD_LOAD(x) _mm256_loadu_ps(x)
#define SIMD_SET(x) _mm256_set1_ps(x)
#define SIMD_MUL(x, y) _mm256_mul_ps(x, y)
#define SIMD_FMA(x, y, c) _mm256_fmadd_ps(x, y, c)
#define SIMD_SQRT(x) _mm256_sqrt_ps(x)
#define SIMD_DIV(x, y) _mm256_div_ps(x, y)
#define SIMD_WIDTH 8
#endif
#endif

class Adam_Optimizer {
public:
    Adam_Optimizer(float alpha = 1e-3,
                   float betta1 = 0.9,
                   float betta2 = 0.999,
                   float eps = 1e-8,
                   float weight_decay = 0,
                   bool adamw_mode = true)
        : _alpha(alpha),
          _betta1(betta1),
          _betta2(betta2),
          _eps(eps),
          _weight_decay(weight_decay),
          _betta1_t(1.0),
          _betta2_t(1.0),
          _step(0),
          _buf_index(false),
          _adamw_mode(adamw_mode)
    {
        cudaMallocHost((void**)_doubled_buffer, TILE * sizeof(float));
        cudaMallocHost((void**)(_doubled_buffer + 1), TILE * sizeof(float));

        _streams[0] = Context::Instance().GetCurrentStream();
        _streams[1] = Context::Instance().GetNewStream();
    }
    ~Adam_Optimizer()
    {
        cudaFreeHost(_doubled_buffer[0]);
        cudaFreeHost(_doubled_buffer[1]);
    }
    void Step(float* _params,
              float* grads,
              float* _exp_avg,
              float* _exp_avg_sq,
              size_t param_size,
              __half* dev_param = nullptr);
    void Step_4(float* _params,
                float* grads,
                float* _exp_avg,
                float* _exp_avg_sa,
                size_t param_size,
                __half* dev_param = nullptr);
    void Step_8(float* _params,
                float* grads,
                float* _exp_avg,
                float* _exp_avg_sq,
                size_t _param_size,
                __half* dev_params = nullptr);
    inline void SynchronizeStreams()
    {
        for (int i = 0; i < 2; i++) cudaStreamSynchronize(_streams[i]);
    }
    inline void IncrementStep(size_t step, float beta1, float beta2)
    {
        if (beta1 != _betta1 || beta2 != _betta2) {
            _step = step;
            _betta1 = beta1;
            _betta2 = beta2;
            _betta1_t = std::pow(_betta1, step);
            _betta2_t = std::pow(_betta2, step);
        } else {
            _step++;
            if (_step != step) {
                _betta1_t = std::pow(_betta1, step);
                _betta2_t = std::pow(_betta2, step);
                _step = step;
            } else {
                _betta1_t *= _betta1;
                _betta2_t *= _betta2;
            }
        }
    }
    inline void update_state(float lr, float epsilon, float weight_decay, bool bias_correction)
    {
        _alpha = lr;
        _eps = epsilon;
        _weight_decay = weight_decay;

        _bias_correction1 = 1.0f;
        _bias_correction2 = 1.0f;
        if (bias_correction == 1) {
            _bias_correction1 = 1 - _betta1_t;
            _bias_correction2 = 1 / sqrt(1 - _betta2_t);
        }
    }

private:
#if defined(__AVX512__) or defined(__AVX256__)
    union AVX_Data {
#if defined(__AVX512__)
        __m512 data;
#else
        __m256 data;
#endif
        // float data_f[16];
    };
#endif

    float _alpha;
    float _betta1;
    float _betta2;
    float _eps;
    float _weight_decay;

    float _betta1_t;
    float _betta2_t;
    size_t _step;

    float _bias_correction1;
    float _bias_correction2;

    float* _doubled_buffer[2];
    bool _buf_index;
    bool _adamw_mode;

    cudaStream_t _streams[2];
};
