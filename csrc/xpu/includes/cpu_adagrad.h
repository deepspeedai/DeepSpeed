// Copyright (c) Microsoft Corporation.
// SPDX-License-Identifier: Apache-2.0

// DeepSpeed Team

#pragma once

#define NOMINMAX  // Windows idiosyncrasy
                  // https://stackoverflow.com/questions/4913922/possible-problems-with-nominmax-on-visual-c

#include <stdio.h>
#include <cassert>
#include "simd.h"

typedef unsigned short ds_half_precision_t;

#define STEP(SPAN)                                             \
    void Step_##SPAN(float* _params,                           \
                     float* grads,                             \
                     float* _exp_avg_sq,                       \
                     size_t _param_size,                       \
                     ds_half_precision_t* dev_param = nullptr, \
                     bool half_precision = false);

class Adagrad_Optimizer {
public:
    Adagrad_Optimizer(float alpha = 1e-2, float eps = 1e-8, float weight_decay = 0)
        : _alpha(alpha), _eps(eps), _weight_decay(weight_decay)
    {
    }
    ~Adagrad_Optimizer() {}
#if defined(__AVX512__) or defined(__AVX256__)
    template <int span>
    void Step_AVX(size_t* rounded_size,
                  float* _params,
                  float* grads,
                  float* _exp_avg_sq,
                  size_t param_size,
                  ds_half_precision_t* dev_param = nullptr,
                  bool half_precision = false);
#endif
    STEP(1)
    STEP(4)
    STEP(8)
    inline void IncrementStep(size_t step)
    {
        _step++;
        if (_step != step) { _step = step; }
    }
    inline void update_state(float lr, float epsilon, float weight_decay)
    {
        _alpha = lr;
        _eps = epsilon;
        _weight_decay = weight_decay;
    }

private:
    float _alpha;
    float _eps;
    float _weight_decay;

    float _betta1_t;
    float _betta2_t;
    size_t _step;
};

#if defined(__AVX512__) or defined(__AVX256__)
template <int span>
void Adagrad_Optimizer::Step_AVX(size_t* rounded_size,
                                 float* _params,
                                 float* grads,
                                 float* _exp_avg_sq,
                                 size_t _param_size,
                                 ds_half_precision_t* dev_params,
                                 bool half_precision)
{
    size_t new_rounded_size = 0;
    AVX_Data eps_4;
    eps_4.data = SIMD_SET(_eps);

    float step_size = -1 * _alpha;
    AVX_Data step_size_4;
    step_size_4.data = SIMD_SET(step_size);

    AVX_Data weight_decay4;
    if (_weight_decay > 0) weight_decay4.data = SIMD_SET(_weight_decay);
    new_rounded_size = ROUND_DOWN(_param_size, SIMD_WIDTH * span);
    for (size_t t = 0; t < new_rounded_size; t += TILE) {
        size_t copy_size = TILE;
        if ((t + TILE) > new_rounded_size) copy_size = new_rounded_size - t;
        size_t offset = copy_size + t;
#pragma omp parallel for
        for (size_t i = t; i < offset; i += SIMD_WIDTH * span) {
            AVX_Data grad_4[span];
            simd_load<span>(grad_4, grads + i, half_precision);

            AVX_Data momentum_4[span];
            simd_load<span>(momentum_4, grads + i, false);

            AVX_Data variance_4[span];
            simd_load<span>(variance_4, _exp_avg_sq + i, false);

            AVX_Data param_4[span];
            simd_load<span>(param_4, _params + i, half_precision);

            if (_weight_decay > 0) { simd_fma<span>(grad_4, param_4, weight_decay4, grad_4); }

            simd_fma<span>(variance_4, grad_4, grad_4, variance_4);
            simd_sqrt<span>(grad_4, variance_4);
            simd_add<span>(grad_4, grad_4, eps_4);
            simd_div<span>(grad_4, momentum_4, grad_4);
            simd_fma<span>(param_4, grad_4, step_size_4, param_4);

            simd_store<span>(_params + i, param_4, half_precision);
            simd_store<span>(_exp_avg_sq + i, variance_4, false);
        }
    }
    *rounded_size = new_rounded_size;
}
#endif
