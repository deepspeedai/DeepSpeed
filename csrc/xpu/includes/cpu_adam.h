// Copyright (c) Microsoft Corporation.
// SPDX-License-Identifier: Apache-2.0

// DeepSpeed Team

#pragma once

#define NOMINMAX  // Windows idiosyncrasy
                  // https://stackoverflow.com/questions/4913922/possible-problems-with-nominmax-on-visual-c

#include <stdio.h>
#include <torch/extension.h>
#include <cassert>
#include "simd.h"

#include <cmath>
typedef unsigned short ds_half_precision_t;

#define STEP(SPAN)                                             \
    void Step_##SPAN(float* _params,                           \
                     float* grads,                             \
                     float* _exp_avg,                          \
                     float* _exp_avg_sq,                       \
                     size_t _param_size,                       \
                     ds_half_precision_t* dev_param = nullptr, \
                     bool half_precision = false);

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
          _adamw_mode(adamw_mode)
    {
    }
    ~Adam_Optimizer() {}

#if defined(__AVX512__) or defined(__AVX256__)
    template <int span>
    void Step_AVX(size_t* rounded_size,
                  float* _params,
                  float* grads,
                  float* _exp_avg,
                  float* _exp_avg_sq,
                  size_t param_size,
                  ds_half_precision_t* dev_param = nullptr,
                  bool half_precision = false);
#endif
    STEP(1)
    STEP(4)
    STEP(8)
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

    bool _adamw_mode;
};

#if defined(__AVX512__) or defined(__AVX256__)
template <int span>
void Adam_Optimizer::Step_AVX(size_t* rounded_size,
                              float* _params,
                              float* grads,
                              float* _exp_avg,
                              float* _exp_avg_sq,
                              size_t _param_size,
                              ds_half_precision_t* dev_params,
                              bool half_precision)
{
    size_t new_rounded_size = 0;
    int rshft = half_precision ? 1 : 0;

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
    new_rounded_size = ROUND_DOWN(_param_size, SIMD_WIDTH * span);
    for (size_t t = 0; t < new_rounded_size; t += TILE) {
        size_t copy_size = TILE;
        if ((t + TILE) > new_rounded_size) copy_size = new_rounded_size - t;
        size_t offset = copy_size + t;
#pragma omp parallel for
        for (size_t i = t; i < offset; i += SIMD_WIDTH * span) {
            AVX_Data grad_4[span];
            simd_load<span>(grad_4, grads + (i >> rshft), half_precision);

            AVX_Data momentum_4[span];
            simd_load<span>(momentum_4, _exp_avg + i, false);

            AVX_Data variance_4[span];
            simd_load<span>(variance_4, _exp_avg_sq + i, false);

            AVX_Data param_4[span];
            simd_load<span>(param_4, _params + (i >> rshft), half_precision);

            if (_weight_decay > 0 && !_adamw_mode) {
                simd_fma<span>(grad_4, param_4, weight_decay4, grad_4);
            }

            simd_mul<span>(momentum_4, momentum_4, betta1_4);
            simd_fma<span>(momentum_4, grad_4, betta1_minus1_4, momentum_4);
            simd_mul<span>(variance_4, variance_4, betta2_4);
            simd_mul<span>(grad_4, grad_4, grad_4);
            simd_fma<span>(variance_4, grad_4, betta2_minus1_4, variance_4);
            simd_sqrt<span>(grad_4, variance_4);
            simd_fma<span>(grad_4, grad_4, bias2_sqrt, eps_4);
            simd_div<span>(grad_4, momentum_4, grad_4);

            if (_weight_decay > 0 && _adamw_mode) {
                simd_fma<span>(param_4, param_4, weight_decay4, param_4);
            }

            simd_fma<span>(param_4, grad_4, step_size_4, param_4);

            simd_store<span>(_params + (i >> rshft), param_4, half_precision);
            simd_store<span>(_exp_avg + i, momentum_4, false);
            simd_store<span>(_exp_avg_sq + i, variance_4, false);
        }
    }
    *rounded_size = new_rounded_size;
}
#endif

int create_adam_optimizer(int optimizer_id,
                          float alpha = 1e-3,
                          float betta1 = 0.9,
                          float betta2 = 0.999,
                          float eps = 1e-8,
                          float weight_decay = 0,
                          bool adamw_mode = true,
                          bool should_log = false);

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
                 torch::Tensor& exp_avg_sq);

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
                           torch::Tensor& gpu_params);

int destroy_adam_optimizer(int optimizer_id);
