#include "cpu_adam.h"
#include <cuda_runtime_api.h>
#include <math.h>
#include <omp.h>
#include <torch/extension.h>
#include <iostream>
#include <memory>
#include <type_traits>
#include <unordered_map>
#include "cublas_v2.h"
#include "cuda.h"
#include "curand.h"
#include "custom_cuda_layers.h"

static std::unordered_map<int, std::shared_ptr<void>> s_optimizers;

#define ROUND_DOWN(size, step) ((size) & ~((step)-1))

// C++ interface

void Adam_Optimizer::Step(float* _params,
                          float* grads,
                          float* _exp_avg,
                          float* _exp_avg_sq,
                          size_t _param_size,
                          __half* dev_params)
{
    float betta1_minus1 = 1 - _betta1;
    float betta2_minus1 = 1 - _betta2;

    float step_size = -1 * _alpha / _bias_correction1;
    float w_decay = -1 * _alpha * _weight_decay;
    size_t rounded_size = 0;

#if defined(__AVX512__) or defined(__AVX256__)

    AVX_Data betta1_4;
    betta1_4.data = SIMD_SET(_betta1);
    AVX_Data betta2_4;
    betta2_4.data = SIMD_SET(_betta2);

    AVX_Data betta1_minus1_4;
    betta1_minus1_4.data = SIMD_SET(betta1_minus1);
    AVX_Data betta2_minus1_4;
    betta2_minus1_4.data = SIMD_SET(betta2_minus1);

    AVX_Data bias2_sqrt;
    bias2_sqrt.data = SIMD_SET(_bias_correction2);

    AVX_Data eps_4;
    eps_4.data = SIMD_SET(_eps);

    AVX_Data step_size_4;
    step_size_4.data = SIMD_SET(step_size);

    AVX_Data weight_decay4;
    if (_weight_decay > 0)
        weight_decay4.data = (_adamw_mode ? SIMD_SET(w_decay) : SIMD_SET(_weight_decay));
    rounded_size = ROUND_DOWN(_param_size, SIMD_WIDTH);

    for (size_t t = 0; t < rounded_size; t += TILE) {
        size_t copy_size = TILE;
        if ((t + TILE) > rounded_size) copy_size = rounded_size - t;
        size_t offset = copy_size + t;
        if ((t / TILE) >= 2) { cudaStreamSynchronize(_streams[_buf_index]); }

#pragma omp parallel for
        for (size_t i = t; i < offset; i += SIMD_WIDTH) {
            AVX_Data grad_4;
            grad_4.data = SIMD_LOAD(grads + i);

            AVX_Data momentum_4;
            momentum_4.data = SIMD_LOAD(_exp_avg + i);
            AVX_Data variance_4;
            variance_4.data = SIMD_LOAD(_exp_avg_sq + i);

            AVX_Data param_4;
            param_4.data = SIMD_LOAD(_params + i);

            if (_weight_decay > 0 && !_adamw_mode) {
                grad_4.data = SIMD_FMA(param_4.data, weight_decay4.data, grad_4.data);
            }
            momentum_4.data = SIMD_MUL(momentum_4.data, betta1_4.data);
            momentum_4.data = SIMD_FMA(grad_4.data, betta1_minus1_4.data, momentum_4.data);

            variance_4.data = SIMD_MUL(variance_4.data, betta2_4.data);
            grad_4.data = SIMD_MUL(grad_4.data, grad_4.data);
            variance_4.data = SIMD_FMA(grad_4.data, betta2_minus1_4.data, variance_4.data);

            grad_4.data = SIMD_SQRT(variance_4.data);
            grad_4.data = SIMD_FMA(grad_4.data, bias2_sqrt.data, eps_4.data);
            grad_4.data = SIMD_DIV(momentum_4.data, grad_4.data);
            if (_weight_decay > 0 && _adamw_mode) {
                param_4.data = SIMD_FMA(param_4.data, weight_decay4.data, param_4.data);
            }
            param_4.data = SIMD_FMA(grad_4.data, step_size_4.data, param_4.data);

            SIMD_STORE(_params + i, param_4.data);

            if (dev_params) SIMD_STORE(_doubled_buffer[_buf_index] + (i - t), param_4.data);

            SIMD_STORE(_exp_avg + i, momentum_4.data);
            SIMD_STORE(_exp_avg_sq + i, variance_4.data);
        }
        if (dev_params) {
            launch_param_update(
                _doubled_buffer[_buf_index], dev_params + t, copy_size, _streams[_buf_index]);
            _buf_index = !_buf_index;
        }
    }

#endif

    if (_param_size > rounded_size) {
        for (size_t t = rounded_size; t < _param_size; t += TILE) {
            size_t copy_size = TILE;
            if ((t + TILE) > _param_size) copy_size = _param_size - t;
            size_t offset = copy_size + t;
            if ((t / TILE) >= 2) { cudaStreamSynchronize(_streams[_buf_index]); }
#pragma omp parallel for
            for (size_t k = t; k < offset; k++) {
                float grad = grads[k];
                float param = _params[k];
                float momentum = _exp_avg[k];
                float variance = _exp_avg_sq[k];
                if (_weight_decay > 0 && !_adamw_mode) { grad = param * _weight_decay + grad; }
                momentum = momentum * _betta1;
                momentum = grad * betta1_minus1 + momentum;

                variance = variance * _betta2;
                grad = grad * grad;
                variance = grad * betta2_minus1 + variance;

                grad = sqrt(variance);
                grad = grad * _bias_correction2 + _eps;
                grad = momentum / grad;
                if (_weight_decay > 0 && _adamw_mode) { param += w_decay * param; }
                param = grad * step_size + param;
                if (dev_params) _doubled_buffer[_buf_index][k - t] = param;

                _params[k] = param;
                _exp_avg[k] = momentum;
                _exp_avg_sq[k] = variance;
            }
            if (dev_params) {
                launch_param_update(
                    _doubled_buffer[_buf_index], dev_params + t, (copy_size), _streams[_buf_index]);
                _buf_index = !_buf_index;
            }
        }
    }
}

void Adam_Optimizer::Step_4(float* _params,
                            float* grads,
                            float* _exp_avg,
                            float* _exp_avg_sq,
                            size_t _param_size,
                            __half* dev_params)
{
    size_t rounded_size = 0;

#if defined(__AVX512__) or defined(__AVX256__)

    AVX_Data betta1_4;
    betta1_4.data = SIMD_SET(_betta1);
    AVX_Data betta2_4;
    betta2_4.data = SIMD_SET(_betta2);

    float betta1_minus1 = 1 - _betta1;
    float betta2_minus1 = 1 - _betta2;
    AVX_Data betta1_minus1_4;
    betta1_minus1_4.data = SIMD_SET(betta1_minus1);
    AVX_Data betta2_minus1_4;
    betta2_minus1_4.data = SIMD_SET(betta2_minus1);

    AVX_Data bias2_sqrt;
    bias2_sqrt.data = SIMD_SET(_bias_correction2);

    AVX_Data eps_4;
    eps_4.data = SIMD_SET(_eps);

    float step_size = -1 * _alpha / _bias_correction1;
    AVX_Data step_size_4;
    step_size_4.data = SIMD_SET(step_size);

    float w_decay = -1 * _alpha * _weight_decay;
    AVX_Data weight_decay4;
    if (_weight_decay > 0)
        weight_decay4.data = (_adamw_mode ? SIMD_SET(w_decay) : SIMD_SET(_weight_decay));
    rounded_size = ROUND_DOWN(_param_size, (SIMD_WIDTH << 2));

    for (size_t t = 0; t < rounded_size; t += TILE) {
        size_t copy_size = TILE;
        if ((t + TILE) > rounded_size) copy_size = rounded_size - t;
        size_t offset = copy_size + t;
        if ((t / TILE) >= 2) { cudaStreamSynchronize(_streams[_buf_index]); }
#pragma omp parallel for
        for (size_t i = t; i < offset; i += (SIMD_WIDTH << 2)) {
            AVX_Data grad_4[4];
            grad_4[0].data = SIMD_LOAD(grads + i);
            grad_4[1].data = SIMD_LOAD(grads + i + SIMD_WIDTH);
            grad_4[2].data = SIMD_LOAD(grads + i + (SIMD_WIDTH << 1));
            grad_4[3].data = SIMD_LOAD(grads + i + SIMD_WIDTH * 3);

            AVX_Data momentum_4[4];
            momentum_4[0].data = SIMD_LOAD(_exp_avg + i);
            momentum_4[1].data = SIMD_LOAD(_exp_avg + i + SIMD_WIDTH);
            momentum_4[2].data = SIMD_LOAD(_exp_avg + i + (SIMD_WIDTH << 1));
            momentum_4[3].data = SIMD_LOAD(_exp_avg + i + SIMD_WIDTH * 3);

            AVX_Data variance_4[4];
            variance_4[0].data = SIMD_LOAD(_exp_avg_sq + i);
            variance_4[1].data = SIMD_LOAD(_exp_avg_sq + i + SIMD_WIDTH);
            variance_4[2].data = SIMD_LOAD(_exp_avg_sq + i + (SIMD_WIDTH << 1));
            variance_4[3].data = SIMD_LOAD(_exp_avg_sq + i + SIMD_WIDTH * 3);

            AVX_Data param_4[4];
            param_4[0].data = SIMD_LOAD(_params + i);
            param_4[1].data = SIMD_LOAD(_params + i + SIMD_WIDTH);
            param_4[2].data = SIMD_LOAD(_params + i + (SIMD_WIDTH << 1));
            param_4[3].data = SIMD_LOAD(_params + i + SIMD_WIDTH * 3);

            if (_weight_decay > 0 && !_adamw_mode) {
                grad_4[0].data = SIMD_FMA(param_4[0].data, weight_decay4.data, grad_4[0].data);
                grad_4[1].data = SIMD_FMA(param_4[1].data, weight_decay4.data, grad_4[1].data);
                grad_4[2].data = SIMD_FMA(param_4[2].data, weight_decay4.data, grad_4[2].data);
                grad_4[3].data = SIMD_FMA(param_4[3].data, weight_decay4.data, grad_4[3].data);
            }

            momentum_4[0].data = SIMD_MUL(momentum_4[0].data, betta1_4.data);
            momentum_4[0].data = SIMD_FMA(grad_4[0].data, betta1_minus1_4.data, momentum_4[0].data);
            momentum_4[1].data = SIMD_MUL(momentum_4[1].data, betta1_4.data);
            momentum_4[1].data = SIMD_FMA(grad_4[1].data, betta1_minus1_4.data, momentum_4[1].data);
            momentum_4[2].data = SIMD_MUL(momentum_4[2].data, betta1_4.data);
            momentum_4[2].data = SIMD_FMA(grad_4[2].data, betta1_minus1_4.data, momentum_4[2].data);
            momentum_4[3].data = SIMD_MUL(momentum_4[3].data, betta1_4.data);
            momentum_4[3].data = SIMD_FMA(grad_4[3].data, betta1_minus1_4.data, momentum_4[3].data);

            variance_4[0].data = SIMD_MUL(variance_4[0].data, betta2_4.data);
            variance_4[1].data = SIMD_MUL(variance_4[1].data, betta2_4.data);
            variance_4[2].data = SIMD_MUL(variance_4[2].data, betta2_4.data);
            variance_4[3].data = SIMD_MUL(variance_4[3].data, betta2_4.data);
            grad_4[0].data = SIMD_MUL(grad_4[0].data, grad_4[0].data);
            grad_4[1].data = SIMD_MUL(grad_4[1].data, grad_4[1].data);
            grad_4[2].data = SIMD_MUL(grad_4[2].data, grad_4[2].data);
            grad_4[3].data = SIMD_MUL(grad_4[3].data, grad_4[3].data);
            variance_4[0].data = SIMD_FMA(grad_4[0].data, betta2_minus1_4.data, variance_4[0].data);
            variance_4[1].data = SIMD_FMA(grad_4[1].data, betta2_minus1_4.data, variance_4[1].data);
            variance_4[2].data = SIMD_FMA(grad_4[2].data, betta2_minus1_4.data, variance_4[2].data);
            variance_4[3].data = SIMD_FMA(grad_4[3].data, betta2_minus1_4.data, variance_4[3].data);

            grad_4[0].data = SIMD_SQRT(variance_4[0].data);
            grad_4[1].data = SIMD_SQRT(variance_4[1].data);
            grad_4[2].data = SIMD_SQRT(variance_4[2].data);
            grad_4[3].data = SIMD_SQRT(variance_4[3].data);

            grad_4[0].data = SIMD_FMA(grad_4[0].data, bias2_sqrt.data, eps_4.data);
            grad_4[1].data = SIMD_FMA(grad_4[1].data, bias2_sqrt.data, eps_4.data);
            grad_4[2].data = SIMD_FMA(grad_4[2].data, bias2_sqrt.data, eps_4.data);
            grad_4[3].data = SIMD_FMA(grad_4[3].data, bias2_sqrt.data, eps_4.data);
            grad_4[0].data = SIMD_DIV(momentum_4[0].data, grad_4[0].data);
            grad_4[1].data = SIMD_DIV(momentum_4[1].data, grad_4[1].data);
            grad_4[2].data = SIMD_DIV(momentum_4[2].data, grad_4[2].data);
            grad_4[3].data = SIMD_DIV(momentum_4[3].data, grad_4[3].data);

            if (_weight_decay > 0 && _adamw_mode) {
                param_4[0].data = SIMD_FMA(param_4[0].data, weight_decay4.data, param_4[0].data);
                param_4[1].data = SIMD_FMA(param_4[1].data, weight_decay4.data, param_4[1].data);
                param_4[2].data = SIMD_FMA(param_4[2].data, weight_decay4.data, param_4[2].data);
                param_4[3].data = SIMD_FMA(param_4[3].data, weight_decay4.data, param_4[3].data);
            }

            param_4[0].data = SIMD_FMA(grad_4[0].data, step_size_4.data, param_4[0].data);
            param_4[1].data = SIMD_FMA(grad_4[1].data, step_size_4.data, param_4[1].data);
            param_4[2].data = SIMD_FMA(grad_4[2].data, step_size_4.data, param_4[2].data);
            param_4[3].data = SIMD_FMA(grad_4[3].data, step_size_4.data, param_4[3].data);

            SIMD_STORE(_params + i, param_4[0].data);
            SIMD_STORE(_params + i + SIMD_WIDTH, param_4[1].data);
            SIMD_STORE(_params + i + (SIMD_WIDTH << 1), param_4[2].data);
            SIMD_STORE(_params + i + SIMD_WIDTH * 3, param_4[3].data);

            if (dev_params) {
                SIMD_STORE(_doubled_buffer[_buf_index] + (i - t), param_4[0].data);
                SIMD_STORE(_doubled_buffer[_buf_index] + (i - t) + SIMD_WIDTH, param_4[1].data);
                SIMD_STORE(_doubled_buffer[_buf_index] + (i - t) + (SIMD_WIDTH << 1),
                           param_4[2].data);
                SIMD_STORE(_doubled_buffer[_buf_index] + (i - t) + SIMD_WIDTH * 3, param_4[3].data);
            }

            SIMD_STORE(_exp_avg + i, momentum_4[0].data);
            SIMD_STORE(_exp_avg + i + SIMD_WIDTH, momentum_4[1].data);
            SIMD_STORE(_exp_avg + i + (SIMD_WIDTH << 1), momentum_4[2].data);
            SIMD_STORE(_exp_avg + i + SIMD_WIDTH * 3, momentum_4[3].data);

            SIMD_STORE(_exp_avg_sq + i, variance_4[0].data);
            SIMD_STORE(_exp_avg_sq + i + SIMD_WIDTH, variance_4[1].data);
            SIMD_STORE(_exp_avg_sq + i + (SIMD_WIDTH << 1), variance_4[2].data);
            SIMD_STORE(_exp_avg_sq + i + SIMD_WIDTH * 3, variance_4[3].data);
        }

        if (dev_params) {
            launch_param_update(
                _doubled_buffer[_buf_index], dev_params + t, copy_size, _streams[_buf_index]);
            _buf_index = !_buf_index;
        }
    }
#endif
    if (_param_size > rounded_size)
        Step((_params + rounded_size),
             (grads + rounded_size),
             (_exp_avg + rounded_size),
             (_exp_avg_sq + rounded_size),
             (_param_size - rounded_size),
             (dev_params != nullptr ? (dev_params + rounded_size) : dev_params));
}

int create_adam_optimizer(int optimizer_id,
                          float alpha = 1e-3,
                          float betta1 = 0.9,
                          float betta2 = 0.999,
                          float eps = 1e-8,
                          float weight_decay = 0,
                          bool adamw_mode = true,
                          bool should_log = false)
{
    auto opt =
        std::make_shared<Adam_Optimizer>(alpha, betta1, betta2, eps, weight_decay, adamw_mode);

    s_optimizers[optimizer_id] = opt;

    if (should_log) {
        std::string avx_type = "";
#if defined(__AVX512__)
        avx_type = "AVX512";
#else
#if defined(__AVX256__)
        avx_type = "AVX2";
#else
        avx_type = "scalar";
#endif
#endif

        printf("Adam Optimizer #%d is created with %s arithmetic capability.\n",
               optimizer_id,
               avx_type.c_str());
        printf("Config: alpha=%f, betas=(%f, %f), weight_decay=%f, adam_w=%d\n",
               alpha,
               betta1,
               betta2,
               weight_decay,
               (int)adamw_mode);
    }

    return 0;
}

void Adam_Optimizer::Step_8(float* _params,
                            float* grads,
                            float* _exp_avg,
                            float* _exp_avg_sq,
                            size_t _param_size,
                            __half* dev_params)
{
    size_t rounded_size = 0;

#if defined(__AVX512__) or defined(__AVX256__)

    AVX_Data betta1_4;
    betta1_4.data = SIMD_SET(_betta1);
    AVX_Data betta2_4;
    betta2_4.data = SIMD_SET(_betta2);

    float betta1_minus1 = 1 - _betta1;
    float betta2_minus1 = 1 - _betta2;
    AVX_Data betta1_minus1_4;
    betta1_minus1_4.data = SIMD_SET(betta1_minus1);
    AVX_Data betta2_minus1_4;
    betta2_minus1_4.data = SIMD_SET(betta2_minus1);

    AVX_Data bias2_sqrt;
    bias2_sqrt.data = SIMD_SET(_bias_correction2);

    AVX_Data eps_4;
    eps_4.data = SIMD_SET(_eps);

    float step_size = -1 * _alpha / _bias_correction1;
    AVX_Data step_size_4;
    step_size_4.data = SIMD_SET(step_size);

    float w_decay = -1 * _alpha * _weight_decay;
    AVX_Data weight_decay4;
    if (_weight_decay > 0)
        weight_decay4.data = (_adamw_mode ? SIMD_SET(w_decay) : SIMD_SET(_weight_decay));
    rounded_size = ROUND_DOWN(_param_size, (SIMD_WIDTH << 3));

    for (size_t t = 0; t < rounded_size; t += TILE) {
        size_t copy_size = TILE;
        if ((t + TILE) > rounded_size) copy_size = rounded_size - t;
        size_t offset = copy_size + t;
        if ((t / TILE) >= 2) { cudaStreamSynchronize(_streams[_buf_index]); }
#pragma omp parallel for
        for (size_t i = t; i < offset; i += (SIMD_WIDTH << 3)) {
            AVX_Data grad_4[8];
            grad_4[0].data = SIMD_LOAD(grads + i);
            grad_4[1].data = SIMD_LOAD(grads + i + SIMD_WIDTH);
            grad_4[2].data = SIMD_LOAD(grads + i + (SIMD_WIDTH << 1));
            grad_4[3].data = SIMD_LOAD(grads + i + SIMD_WIDTH * 3);
            grad_4[4].data = SIMD_LOAD(grads + i + (SIMD_WIDTH << 2));
            grad_4[5].data = SIMD_LOAD(grads + i + SIMD_WIDTH * 5);
            grad_4[6].data = SIMD_LOAD(grads + i + SIMD_WIDTH * 6);
            grad_4[7].data = SIMD_LOAD(grads + i + SIMD_WIDTH * 7);

            AVX_Data momentum_4[8];
            momentum_4[0].data = SIMD_LOAD(_exp_avg + i);
            momentum_4[1].data = SIMD_LOAD(_exp_avg + i + SIMD_WIDTH);
            momentum_4[2].data = SIMD_LOAD(_exp_avg + i + (SIMD_WIDTH << 1));
            momentum_4[3].data = SIMD_LOAD(_exp_avg + i + SIMD_WIDTH * 3);
            momentum_4[4].data = SIMD_LOAD(_exp_avg + i + (SIMD_WIDTH << 2));
            momentum_4[5].data = SIMD_LOAD(_exp_avg + i + SIMD_WIDTH * 5);
            momentum_4[6].data = SIMD_LOAD(_exp_avg + i + SIMD_WIDTH * 6);
            momentum_4[7].data = SIMD_LOAD(_exp_avg + i + SIMD_WIDTH * 7);

            AVX_Data variance_4[8];
            variance_4[0].data = SIMD_LOAD(_exp_avg_sq + i);
            variance_4[1].data = SIMD_LOAD(_exp_avg_sq + i + SIMD_WIDTH);
            variance_4[2].data = SIMD_LOAD(_exp_avg_sq + i + (SIMD_WIDTH << 1));
            variance_4[3].data = SIMD_LOAD(_exp_avg_sq + i + SIMD_WIDTH * 3);
            variance_4[4].data = SIMD_LOAD(_exp_avg_sq + i + (SIMD_WIDTH << 2));
            variance_4[5].data = SIMD_LOAD(_exp_avg_sq + i + SIMD_WIDTH * 5);
            variance_4[6].data = SIMD_LOAD(_exp_avg_sq + i + SIMD_WIDTH * 6);
            variance_4[7].data = SIMD_LOAD(_exp_avg_sq + i + SIMD_WIDTH * 7);

            AVX_Data param_4[8];
            param_4[0].data = SIMD_LOAD(_params + i);
            param_4[1].data = SIMD_LOAD(_params + i + SIMD_WIDTH);
            param_4[2].data = SIMD_LOAD(_params + i + (SIMD_WIDTH << 1));
            param_4[3].data = SIMD_LOAD(_params + i + SIMD_WIDTH * 3);
            param_4[4].data = SIMD_LOAD(_params + i + (SIMD_WIDTH << 2));
            param_4[5].data = SIMD_LOAD(_params + i + SIMD_WIDTH * 5);
            param_4[6].data = SIMD_LOAD(_params + i + SIMD_WIDTH * 6);
            param_4[7].data = SIMD_LOAD(_params + i + SIMD_WIDTH * 7);

            if (_weight_decay > 0 && !_adamw_mode) {
                grad_4[0].data = SIMD_FMA(param_4[0].data, weight_decay4.data, grad_4[0].data);
                grad_4[1].data = SIMD_FMA(param_4[1].data, weight_decay4.data, grad_4[1].data);
                grad_4[2].data = SIMD_FMA(param_4[2].data, weight_decay4.data, grad_4[2].data);
                grad_4[3].data = SIMD_FMA(param_4[3].data, weight_decay4.data, grad_4[3].data);
                grad_4[4].data = SIMD_FMA(param_4[4].data, weight_decay4.data, grad_4[4].data);
                grad_4[5].data = SIMD_FMA(param_4[5].data, weight_decay4.data, grad_4[5].data);
                grad_4[6].data = SIMD_FMA(param_4[6].data, weight_decay4.data, grad_4[6].data);
                grad_4[7].data = SIMD_FMA(param_4[7].data, weight_decay4.data, grad_4[7].data);
            }

            momentum_4[0].data = SIMD_MUL(momentum_4[0].data, betta1_4.data);
            momentum_4[0].data = SIMD_FMA(grad_4[0].data, betta1_minus1_4.data, momentum_4[0].data);
            momentum_4[1].data = SIMD_MUL(momentum_4[1].data, betta1_4.data);
            momentum_4[1].data = SIMD_FMA(grad_4[1].data, betta1_minus1_4.data, momentum_4[1].data);
            momentum_4[2].data = SIMD_MUL(momentum_4[2].data, betta1_4.data);
            momentum_4[2].data = SIMD_FMA(grad_4[2].data, betta1_minus1_4.data, momentum_4[2].data);
            momentum_4[3].data = SIMD_MUL(momentum_4[3].data, betta1_4.data);
            momentum_4[3].data = SIMD_FMA(grad_4[3].data, betta1_minus1_4.data, momentum_4[3].data);
            momentum_4[4].data = SIMD_MUL(momentum_4[4].data, betta1_4.data);
            momentum_4[4].data = SIMD_FMA(grad_4[4].data, betta1_minus1_4.data, momentum_4[4].data);
            momentum_4[5].data = SIMD_MUL(momentum_4[5].data, betta1_4.data);
            momentum_4[5].data = SIMD_FMA(grad_4[5].data, betta1_minus1_4.data, momentum_4[5].data);
            momentum_4[6].data = SIMD_MUL(momentum_4[6].data, betta1_4.data);
            momentum_4[6].data = SIMD_FMA(grad_4[6].data, betta1_minus1_4.data, momentum_4[6].data);
            momentum_4[7].data = SIMD_MUL(momentum_4[7].data, betta1_4.data);
            momentum_4[7].data = SIMD_FMA(grad_4[7].data, betta1_minus1_4.data, momentum_4[7].data);

            variance_4[0].data = SIMD_MUL(variance_4[0].data, betta2_4.data);
            variance_4[1].data = SIMD_MUL(variance_4[1].data, betta2_4.data);
            variance_4[2].data = SIMD_MUL(variance_4[2].data, betta2_4.data);
            variance_4[3].data = SIMD_MUL(variance_4[3].data, betta2_4.data);
            variance_4[4].data = SIMD_MUL(variance_4[4].data, betta2_4.data);
            variance_4[5].data = SIMD_MUL(variance_4[5].data, betta2_4.data);
            variance_4[6].data = SIMD_MUL(variance_4[6].data, betta2_4.data);
            variance_4[7].data = SIMD_MUL(variance_4[7].data, betta2_4.data);
            grad_4[0].data = SIMD_MUL(grad_4[0].data, grad_4[0].data);
            grad_4[1].data = SIMD_MUL(grad_4[1].data, grad_4[1].data);
            grad_4[2].data = SIMD_MUL(grad_4[2].data, grad_4[2].data);
            grad_4[3].data = SIMD_MUL(grad_4[3].data, grad_4[3].data);
            grad_4[4].data = SIMD_MUL(grad_4[4].data, grad_4[4].data);
            grad_4[5].data = SIMD_MUL(grad_4[5].data, grad_4[5].data);
            grad_4[6].data = SIMD_MUL(grad_4[6].data, grad_4[6].data);
            grad_4[7].data = SIMD_MUL(grad_4[7].data, grad_4[7].data);
            variance_4[0].data = SIMD_FMA(grad_4[0].data, betta2_minus1_4.data, variance_4[0].data);
            variance_4[1].data = SIMD_FMA(grad_4[1].data, betta2_minus1_4.data, variance_4[1].data);
            variance_4[2].data = SIMD_FMA(grad_4[2].data, betta2_minus1_4.data, variance_4[2].data);
            variance_4[3].data = SIMD_FMA(grad_4[3].data, betta2_minus1_4.data, variance_4[3].data);
            variance_4[4].data = SIMD_FMA(grad_4[4].data, betta2_minus1_4.data, variance_4[4].data);
            variance_4[5].data = SIMD_FMA(grad_4[5].data, betta2_minus1_4.data, variance_4[5].data);
            variance_4[6].data = SIMD_FMA(grad_4[6].data, betta2_minus1_4.data, variance_4[6].data);
            variance_4[7].data = SIMD_FMA(grad_4[7].data, betta2_minus1_4.data, variance_4[7].data);

            grad_4[0].data = SIMD_SQRT(variance_4[0].data);
            grad_4[1].data = SIMD_SQRT(variance_4[1].data);
            grad_4[2].data = SIMD_SQRT(variance_4[2].data);
            grad_4[3].data = SIMD_SQRT(variance_4[3].data);
            grad_4[4].data = SIMD_SQRT(variance_4[4].data);
            grad_4[5].data = SIMD_SQRT(variance_4[5].data);
            grad_4[6].data = SIMD_SQRT(variance_4[6].data);
            grad_4[7].data = SIMD_SQRT(variance_4[7].data);

            grad_4[0].data = SIMD_FMA(grad_4[0].data, bias2_sqrt.data, eps_4.data);
            grad_4[1].data = SIMD_FMA(grad_4[1].data, bias2_sqrt.data, eps_4.data);
            grad_4[2].data = SIMD_FMA(grad_4[2].data, bias2_sqrt.data, eps_4.data);
            grad_4[3].data = SIMD_FMA(grad_4[3].data, bias2_sqrt.data, eps_4.data);
            grad_4[4].data = SIMD_FMA(grad_4[4].data, bias2_sqrt.data, eps_4.data);
            grad_4[5].data = SIMD_FMA(grad_4[5].data, bias2_sqrt.data, eps_4.data);
            grad_4[6].data = SIMD_FMA(grad_4[6].data, bias2_sqrt.data, eps_4.data);
            grad_4[7].data = SIMD_FMA(grad_4[7].data, bias2_sqrt.data, eps_4.data);
            grad_4[0].data = SIMD_DIV(momentum_4[0].data, grad_4[0].data);
            grad_4[1].data = SIMD_DIV(momentum_4[1].data, grad_4[1].data);
            grad_4[2].data = SIMD_DIV(momentum_4[2].data, grad_4[2].data);
            grad_4[3].data = SIMD_DIV(momentum_4[3].data, grad_4[3].data);
            grad_4[4].data = SIMD_DIV(momentum_4[4].data, grad_4[4].data);
            grad_4[5].data = SIMD_DIV(momentum_4[5].data, grad_4[5].data);
            grad_4[6].data = SIMD_DIV(momentum_4[6].data, grad_4[6].data);
            grad_4[7].data = SIMD_DIV(momentum_4[7].data, grad_4[7].data);

            if (_weight_decay > 0 && _adamw_mode) {
                param_4[0].data = SIMD_FMA(param_4[0].data, weight_decay4.data, param_4[0].data);
                param_4[1].data = SIMD_FMA(param_4[1].data, weight_decay4.data, param_4[1].data);
                param_4[2].data = SIMD_FMA(param_4[2].data, weight_decay4.data, param_4[2].data);
                param_4[3].data = SIMD_FMA(param_4[3].data, weight_decay4.data, param_4[3].data);
                param_4[4].data = SIMD_FMA(param_4[4].data, weight_decay4.data, param_4[4].data);
                param_4[5].data = SIMD_FMA(param_4[5].data, weight_decay4.data, param_4[5].data);
                param_4[6].data = SIMD_FMA(param_4[6].data, weight_decay4.data, param_4[6].data);
                param_4[7].data = SIMD_FMA(param_4[7].data, weight_decay4.data, param_4[7].data);
            }

            param_4[0].data = SIMD_FMA(grad_4[0].data, step_size_4.data, param_4[0].data);
            param_4[1].data = SIMD_FMA(grad_4[1].data, step_size_4.data, param_4[1].data);
            param_4[2].data = SIMD_FMA(grad_4[2].data, step_size_4.data, param_4[2].data);
            param_4[3].data = SIMD_FMA(grad_4[3].data, step_size_4.data, param_4[3].data);
            param_4[4].data = SIMD_FMA(grad_4[4].data, step_size_4.data, param_4[4].data);
            param_4[5].data = SIMD_FMA(grad_4[5].data, step_size_4.data, param_4[5].data);
            param_4[6].data = SIMD_FMA(grad_4[6].data, step_size_4.data, param_4[6].data);
            param_4[7].data = SIMD_FMA(grad_4[7].data, step_size_4.data, param_4[7].data);

            SIMD_STORE(_params + i, param_4[0].data);
            SIMD_STORE(_params + i + SIMD_WIDTH, param_4[1].data);
            SIMD_STORE(_params + i + (SIMD_WIDTH << 1), param_4[2].data);
            SIMD_STORE(_params + i + SIMD_WIDTH * 3, param_4[3].data);
            SIMD_STORE(_params + i + (SIMD_WIDTH << 2), param_4[4].data);
            SIMD_STORE(_params + i + SIMD_WIDTH * 5, param_4[5].data);
            SIMD_STORE(_params + i + SIMD_WIDTH * 6, param_4[6].data);
            SIMD_STORE(_params + i + SIMD_WIDTH * 7, param_4[7].data);

            if (dev_params) {
                SIMD_STORE(_doubled_buffer[_buf_index] + (i - t), param_4[0].data);
                SIMD_STORE(_doubled_buffer[_buf_index] + (i - t) + SIMD_WIDTH, param_4[1].data);
                SIMD_STORE(_doubled_buffer[_buf_index] + (i - t) + (SIMD_WIDTH << 1),
                           param_4[2].data);
                SIMD_STORE(_doubled_buffer[_buf_index] + (i - t) + SIMD_WIDTH * 3, param_4[3].data);
                SIMD_STORE(_doubled_buffer[_buf_index] + (i - t) + (SIMD_WIDTH << 2),
                           param_4[4].data);
                SIMD_STORE(_doubled_buffer[_buf_index] + (i - t) + SIMD_WIDTH * 5, param_4[5].data);
                SIMD_STORE(_doubled_buffer[_buf_index] + (i - t) + SIMD_WIDTH * 6, param_4[6].data);
                SIMD_STORE(_doubled_buffer[_buf_index] + (i - t) + SIMD_WIDTH * 7, param_4[7].data);
            }

            SIMD_STORE(_exp_avg + i, momentum_4[0].data);
            SIMD_STORE(_exp_avg + i + SIMD_WIDTH, momentum_4[1].data);
            SIMD_STORE(_exp_avg + i + (SIMD_WIDTH << 1), momentum_4[2].data);
            SIMD_STORE(_exp_avg + i + SIMD_WIDTH * 3, momentum_4[3].data);
            SIMD_STORE(_exp_avg + i + (SIMD_WIDTH << 2), momentum_4[4].data);
            SIMD_STORE(_exp_avg + i + SIMD_WIDTH * 5, momentum_4[5].data);
            SIMD_STORE(_exp_avg + i + SIMD_WIDTH * 6, momentum_4[6].data);
            SIMD_STORE(_exp_avg + i + SIMD_WIDTH * 7, momentum_4[7].data);

            SIMD_STORE(_exp_avg_sq + i, variance_4[0].data);
            SIMD_STORE(_exp_avg_sq + i + SIMD_WIDTH, variance_4[1].data);
            SIMD_STORE(_exp_avg_sq + i + (SIMD_WIDTH << 1), variance_4[2].data);
            SIMD_STORE(_exp_avg_sq + i + SIMD_WIDTH * 3, variance_4[3].data);
            SIMD_STORE(_exp_avg_sq + i + (SIMD_WIDTH << 2), variance_4[4].data);
            SIMD_STORE(_exp_avg_sq + i + SIMD_WIDTH * 5, variance_4[5].data);
            SIMD_STORE(_exp_avg_sq + i + SIMD_WIDTH * 6, variance_4[6].data);
            SIMD_STORE(_exp_avg_sq + i + SIMD_WIDTH * 7, variance_4[7].data);
        }
        if (dev_params) {
            launch_param_update(
                _doubled_buffer[_buf_index], dev_params + t, copy_size, _streams[_buf_index]);
            _buf_index = !_buf_index;
        }
    }
#endif
    if (_param_size > rounded_size)
        Step_4((_params + rounded_size),
               (grads + rounded_size),
               (_exp_avg + rounded_size),
               (_exp_avg_sq + rounded_size),
               (_param_size - rounded_size),
               (dev_params != nullptr ? (dev_params + rounded_size) : dev_params));
}

int ds_adam_step(int optimizer_id,
                 size_t step,
                 float lr,
                 float beta1,
                 float beta2,
                 float epsilon,
                 float weight_decay,
                 bool bias_correction,
                 torch::Tensor& params,
                 torch::Tensor& grads,
                 torch::Tensor& exp_avg,
                 torch::Tensor& exp_avg_sq)
{
    auto params_c = params.contiguous();
    auto grads_c = grads.contiguous();
    auto exp_avg_c = exp_avg.contiguous();
    auto exp_avg_sq_c = exp_avg_sq.contiguous();

    float* params_ptr = (float*)params_c.data_ptr();
    float* grads_ptr = (float*)grads_c.data_ptr();
    float* exp_avg_ptr = (float*)exp_avg_c.data_ptr();
    float* exp_avg_sq_ptr = (float*)exp_avg_sq_c.data_ptr();

    std::shared_ptr<Adam_Optimizer> opt =
        std::static_pointer_cast<Adam_Optimizer>(s_optimizers[optimizer_id]);
    opt->IncrementStep(step, beta1, beta2);
    opt->update_state(lr, epsilon, weight_decay, bias_correction);
    opt->Step_8(params_ptr, grads_ptr, exp_avg_ptr, exp_avg_sq_ptr, params_c.size(0));

    opt->SynchronizeStreams();
    return 0;
}

int ds_adam_step_plus_copy(int optimizer_id,
                           size_t step,
                           float lr,
                           float beta1,
                           float beta2,
                           float epsilon,
                           float weight_decay,
                           bool bias_correction,
                           torch::Tensor& params,
                           torch::Tensor& grads,
                           torch::Tensor& exp_avg,
                           torch::Tensor& exp_avg_sq,
                           torch::Tensor& gpu_params)
{
    auto params_c = params.contiguous();
    auto gpu_params_c = gpu_params.contiguous();
    auto exp_avg_c = exp_avg.contiguous();
    auto exp_avg_sq_c = exp_avg_sq.contiguous();
    auto grads_c = grads.contiguous();

    float* params_ptr = (float*)params_c.data_ptr();
    float* grads_ptr = (float*)grads_c.data_ptr();
    __half* gpu_params_ptr = (__half*)gpu_params_c.data_ptr();
    float* exp_avg_ptr = (float*)exp_avg_c.data_ptr();
    float* exp_avg_sq_ptr = (float*)exp_avg_sq_c.data_ptr();

    std::shared_ptr<Adam_Optimizer> opt =
        std::static_pointer_cast<Adam_Optimizer>(s_optimizers[optimizer_id]);
    opt->IncrementStep(step, beta1, beta2);
    opt->update_state(lr, epsilon, weight_decay, bias_correction);
    opt->Step_8(
        params_ptr, grads_ptr, exp_avg_ptr, exp_avg_sq_ptr, params_c.size(0), gpu_params_ptr);

    opt->SynchronizeStreams();
    return 0;
}

int destroy_adam_optimizer(int optimizer_id)
{
    s_optimizers.erase(optimizer_id);

    return 0;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("adam_update", &ds_adam_step, "DeepSpeed CPU Adam update (C++)");
    m.def("adam_update_copy",
          &ds_adam_step_plus_copy,
          "DeepSpeed CPU Adam update and param copy (C++)");
    m.def("create_adam", &create_adam_optimizer, "DeepSpeed CPU Adam (C++)");
    m.def("destroy_adam", &destroy_adam_optimizer, "DeepSpeed CPU Adam destroy (C++)");
}
