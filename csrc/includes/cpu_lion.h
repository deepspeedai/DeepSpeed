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

#define STEP(SPAN)                                                           \
    template <typename ds_params_precision_t, typename ds_state_precision_t> \
    void Step_##SPAN(ds_params_precision_t* _params,                         \
                     ds_params_precision_t* grads,                           \
                     ds_state_precision_t* _exp_avg,                         \
                     size_t _param_size);

class Lion_Optimizer {
public:
    Lion_Optimizer(float alpha = 1e-3,
                   float betta1 = 0.9,
                   float betta2 = 0.999,
                   float weight_decay = 0)
        : _alpha(alpha), _betta1(betta1), _betta2(betta2), _weight_decay(weight_decay), _step(0)
    {
    }
    ~Lion_Optimizer() {}

#if defined(__AVX512__) or defined(__AVX256__)
    template <int span, typename ds_params_precision_t, typename ds_state_precision_t>
    void Step_AVX(size_t* rounded_size,
                  ds_params_precision_t* _params,
                  ds_params_precision_t* grads,
                  ds_state_precision_t* _exp_avg,
                  size_t param_size);
#endif
    STEP(1)
    STEP(4)
    STEP(8)

    inline void IncrementStep(size_t step, float beta1, float beta2)
    {
        _step++;
        if (_step != step || beta1 != _betta1 || beta2 != _betta2) {
            _step = step;
            _betta1 = beta1;
            _betta2 = beta2;
        }
    }
    inline void update_state(float lr, float weight_decay)
    {
        _alpha = lr;
        _weight_decay = weight_decay;
    }

private:
    float _alpha;
    float _betta1;
    float _betta2;
    float _weight_decay;
    size_t _step;
};

#if defined(__AVX512__) or defined(__AVX256__)
template <int span, typename ds_params_precision_t, typename ds_state_precision_t>
void Lion_Optimizer::Step_AVX(size_t* rounded_size,
                              ds_params_precision_t* _params,
                              ds_params_precision_t* grads,
                              ds_state_precision_t* _exp_avg,
                              size_t _param_size)
{
#if !defined(__AVX512__)
    if (std::is_same_v<ds_params_precision_t, c10::BFloat16> ||
        std::is_same_v<ds_state_precision_t, c10::BFloat16>) {
        return;
    }
#endif
    size_t new_rounded_size = 0;

    constexpr float neg1 = -1.0f;
    AVX_Data neg1_4;
    neg1_4.data = SIMD_SET(neg1);

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

    float step_size = -_alpha;
    AVX_Data step_size_4;
    step_size_4.data = SIMD_SET(step_size);

    float after_decay = 1.0f - _alpha * _weight_decay;
    AVX_Data after_decay_4;
    if (_weight_decay > 0) after_decay_4.data = SIMD_SET(after_decay);

    new_rounded_size = ROUND_DOWN(_param_size, SIMD_WIDTH * span);
    for (size_t t = 0; t < new_rounded_size; t += TILE) {
        size_t copy_size = TILE;
        if ((t + TILE) > new_rounded_size) copy_size = new_rounded_size - t;
        size_t offset = copy_size + t;

#pragma omp parallel for
        for (size_t i = t; i < offset; i += SIMD_WIDTH * span) {
            AVX_Data grad_4[span];
            simd_load<span>(grad_4, grads + i);

            AVX_Data momentum_4[span];
            simd_load<span>(momentum_4, _exp_avg + i);

            AVX_Data param_4[span];
            simd_load<span>(param_4, _params + i);

            AVX_Data tmp_4[span];

            simd_mul<span>(tmp_4, momentum_4, betta1_4);
            simd_fma<span>(tmp_4, grad_4, betta1_minus1_4, tmp_4);
            // We already used intrinsics, so consider the machine representation fixed.
            simd_and<span>(tmp_4, tmp_4, neg1_4);
            simd_xor<span>(tmp_4, tmp_4, step_size_4);
            if (_weight_decay > 0) {
                simd_fma<span>(param_4, param_4, after_decay_4, tmp_4);
            } else {
                simd_add<span>(param_4, param_4, tmp_4);
            }

            simd_mul<span>(momentum_4, momentum_4, betta2_4);
            simd_fma<span>(momentum_4, grad_4, betta2_minus1_4, momentum_4);

            simd_store<span>(_params + i, param_4);
            simd_store<span>(_exp_avg + i, momentum_4);
        }
    }
    *rounded_size = new_rounded_size;
}
#endif

int create_lion_optimizer(int optimizer_id,
                          float alpha = 1e-3,
                          float betta1 = 0.9,
                          float betta2 = 0.999,
                          float weight_decay = 0,
                          bool should_log = false);

int ds_lion_step(int optimizer_id,
                 size_t step,
                 float lr,
                 float beta1,
                 float beta2,
                 float weight_decay,
                 torch::Tensor& params,
                 torch::Tensor& grads,
                 torch::Tensor& exp_avg);

int destroy_lion_optimizer(int optimizer_id);
