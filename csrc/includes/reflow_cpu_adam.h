// Copyright (c) Microsoft Corporation.
// SPDX-License-Identifier: Apache-2.0

// DeepSpeed Team

#pragma once

#define NOMINMAX  // Windows idiosyncrasy
                  // https://stackoverflow.com/questions/4913922/possible-problems-with-nominmax-on-visual-c

#if defined(_OPENMP)
#include <omp.h>
#else
/* Builds without OpenMP (e.g. Apple clang without libomp) run the kernels on one thread. */
inline int omp_get_max_threads() { return 1; }
inline int omp_get_num_procs() { return 1; }
inline void omp_set_num_threads(int) {}
#endif
#include <stdio.h>
#include <torch/extension.h>
#include <cassert>
#include <cstdint>
#include "reflow_simd.h"

/* Enable madvise(MADV_HUGEPAGE) hints in the streaming kernels (Linux only).
 * Transparent huge pages reduce TLB pressure for the large, sequentially
 * streamed param/grad/state buffers.
 */
#define USE_HUGE_PAGE 1
#ifdef __linux__
#include <sys/mman.h>
#endif

class Reflow_Adam_Optimizer {
public:
    Reflow_Adam_Optimizer(float alpha = 1e-3,
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
          _bias_correction1(1.0),
          _bias_correction2(1.0),
          _adamw_mode(adamw_mode),
          _num_threads(-1)
    {
    }
    ~Reflow_Adam_Optimizer() {}

#if defined(__AVX512__) or defined(__AVX256__)
    // Adam over the SIMD-aligned prefix of a half-precision gradient. With store_half_params it
    // writes only the new FP16/BF16 params to half_params (param generation); otherwise it commits
    // the FP32 master and the state.
    template <int span,
              typename ds_grad_precision_t,
              typename ds_state_precision_t,
              typename ds_store_precision_t,
              bool store_half_params>
    void Step_AVX_HalfGrad(size_t* rounded_size,
                           float* _params_fp32,
                           ds_grad_precision_t* grads_half,
                           ds_state_precision_t* _exp_avg,
                           ds_state_precision_t* _exp_avg_sq,
                           size_t param_size,
                           ds_store_precision_t* half_params,
                           float combined_scale);

    template <int span,
              typename ds_grad_precision_t,
              typename ds_state_precision_t,
              typename ds_store_precision_t,
              bool unscale_grads,
              bool decay_grad,
              bool decay_param,
              bool store_half_params>
    void Step_AVX_HalfGrad_Loop(size_t rounded_size,
                                float* _params_fp32,
                                ds_grad_precision_t* grads_half,
                                ds_state_precision_t* _exp_avg,
                                ds_state_precision_t* _exp_avg_sq,
                                ds_store_precision_t* half_params,
                                float combined_scale);

#endif
    inline void IncrementStep(size_t step, float beta1, float beta2)
    {
        if (beta1 != _betta1 || beta2 != _betta2) {
            _step = step;
            _betta1 = beta1;
            _betta2 = beta2;
            _betta1_t = std::pow(_betta1, step);
            _betta2_t = std::pow(_betta2, step);
        } else {
            if (step == _step + 1) {  // first optimizer step increase
                _step++;
                _betta1_t *= _betta1;
                _betta2_t *= _betta2;
            } else if (step ==
                       _step) {  // no need to update step; beta1_t and beta2_t already updated
                return;
            } else {  // support step increase not equal to 1
                _betta1_t = std::pow(_betta1, step);
                _betta2_t = std::pow(_betta2, step);
                _step = step;
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

    /* -1 (the default) means "use all available cores"; a positive value caps the
     * OpenMP thread count for the CPU-Adam state update.
     */
    inline void set_num_threads(int num_threads) { _num_threads = num_threads; }
    inline int get_num_threads() const { return _num_threads; }

    // Half/BFloat16 gradient wrapper, params-only FP16/BF16 store.
    template <typename ds_store_precision_t,
              typename ds_grad_precision_t,
              typename ds_state_precision_t>
    void Step_8_FP16_HalfGrad(float* _params_fp32,
                              ds_grad_precision_t* grads_half,
                              ds_state_precision_t* _exp_avg,
                              ds_state_precision_t* _exp_avg_sq,
                              size_t _param_size,
                              ds_store_precision_t* half_params,
                              float combined_scale);

    // Half/BFloat16 gradient wrapper, state-only update.
    template <typename ds_grad_precision_t, typename ds_state_precision_t>
    void Step_8_State_HalfGrad(float* _params_fp32,
                               ds_grad_precision_t* grads_half,
                               ds_state_precision_t* _exp_avg,
                               ds_state_precision_t* _exp_avg_sq,
                               size_t _param_size,
                               float combined_scale = 1.0f);

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

    // OpenMP thread cap for the CPU-Adam state update; -1 means all available cores.
    int _num_threads;
};

#if defined(__AVX512__) or defined(__AVX256__)

/* Half/BFloat16 gradient path: params stay FP32, grads are Half/BFloat16,
 * arithmetic is FP32, results stored to FP32 params and state.
 */
template <int span,
          typename ds_grad_precision_t,
          typename ds_state_precision_t,
          typename ds_store_precision_t,
          bool store_half_params>
void Reflow_Adam_Optimizer::Step_AVX_HalfGrad(size_t* rounded_size,
                                              float* _params_fp32,
                                              ds_grad_precision_t* grads_half,
                                              ds_state_precision_t* _exp_avg,
                                              ds_state_precision_t* _exp_avg_sq,
                                              size_t _param_size,
                                              ds_store_precision_t* half_params,
                                              float combined_scale)
{
    size_t new_rounded_size = ROUND_DOWN(_param_size, SIMD_WIDTH * span);

#if defined(__linux__) && USE_HUGE_PAGE
    if (new_rounded_size > 0) {
        size_t page_size = 4096;
        auto madvise_bytes = [&](size_t elem_size) {
            return (_param_size * elem_size + page_size - 1) & ~(page_size - 1);
        };
        size_t params_bytes = madvise_bytes(sizeof(float));
        size_t grad_bytes = madvise_bytes(sizeof(ds_grad_precision_t));
        size_t state_bytes = madvise_bytes(sizeof(ds_state_precision_t));
        void* params_aligned = (void*)((uintptr_t)_params_fp32 & ~(page_size - 1));
        madvise(params_aligned, params_bytes, MADV_HUGEPAGE);
        madvise((void*)((uintptr_t)grads_half & ~(page_size - 1)), grad_bytes, MADV_HUGEPAGE);
        madvise((void*)((uintptr_t)_exp_avg & ~(page_size - 1)), state_bytes, MADV_HUGEPAGE);
        madvise((void*)((uintptr_t)_exp_avg_sq & ~(page_size - 1)), state_bytes, MADV_HUGEPAGE);
        if (store_half_params) {
            size_t store_bytes = madvise_bytes(sizeof(ds_store_precision_t));
            madvise((void*)((uintptr_t)half_params & ~(page_size - 1)), store_bytes, MADV_HUGEPAGE);
        }
    }
#endif

    // Pick the loop specialized for this step's flags, so no per-element branch remains.
    bool unscale_grads = (combined_scale > 1.0f);
    bool decay_grad = (_weight_decay > 0 && !_adamw_mode);
    bool decay_param = (_weight_decay > 0 && _adamw_mode);
    using G = ds_grad_precision_t;
    using S = ds_state_precision_t;
    using H = ds_store_precision_t;
    if (unscale_grads && decay_grad) {
        Step_AVX_HalfGrad_Loop<span, G, S, H, true, true, false, store_half_params>(
            new_rounded_size,
            _params_fp32,
            grads_half,
            _exp_avg,
            _exp_avg_sq,
            half_params,
            combined_scale);
    } else if (unscale_grads && decay_param) {
        Step_AVX_HalfGrad_Loop<span, G, S, H, true, false, true, store_half_params>(
            new_rounded_size,
            _params_fp32,
            grads_half,
            _exp_avg,
            _exp_avg_sq,
            half_params,
            combined_scale);
    } else if (unscale_grads) {
        Step_AVX_HalfGrad_Loop<span, G, S, H, true, false, false, store_half_params>(
            new_rounded_size,
            _params_fp32,
            grads_half,
            _exp_avg,
            _exp_avg_sq,
            half_params,
            combined_scale);
    } else if (decay_grad) {
        Step_AVX_HalfGrad_Loop<span, G, S, H, false, true, false, store_half_params>(
            new_rounded_size,
            _params_fp32,
            grads_half,
            _exp_avg,
            _exp_avg_sq,
            half_params,
            combined_scale);
    } else if (decay_param) {
        Step_AVX_HalfGrad_Loop<span, G, S, H, false, false, true, store_half_params>(
            new_rounded_size,
            _params_fp32,
            grads_half,
            _exp_avg,
            _exp_avg_sq,
            half_params,
            combined_scale);
    } else {
        Step_AVX_HalfGrad_Loop<span, G, S, H, false, false, false, store_half_params>(
            new_rounded_size,
            _params_fp32,
            grads_half,
            _exp_avg,
            _exp_avg_sq,
            half_params,
            combined_scale);
    }

    *rounded_size = new_rounded_size;
}

// Template flags eliminate per-element branches from each specialized loop.
template <int span,
          typename ds_grad_precision_t,
          typename ds_state_precision_t,
          typename ds_store_precision_t,
          bool unscale_grads,
          bool decay_grad,
          bool decay_param,
          bool store_half_params>
void Reflow_Adam_Optimizer::Step_AVX_HalfGrad_Loop(size_t rounded_size,
                                                   float* _params_fp32,
                                                   ds_grad_precision_t* grads_half,
                                                   ds_state_precision_t* _exp_avg,
                                                   ds_state_precision_t* _exp_avg_sq,
                                                   ds_store_precision_t* half_params,
                                                   float combined_scale)
{
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

    // Adam folds the decay into the gradient; AdamW decays the parameter by lr * weight_decay.
    float w_decay = -1 * _alpha * _weight_decay;
    AVX_Data weight_decay4;
    if (decay_grad) { weight_decay4.data = SIMD_SET(_weight_decay); }
    if (decay_param) { weight_decay4.data = SIMD_SET(w_decay); }

    AVX_Data unscale_factor;
    if (unscale_grads) { unscale_factor.data = SIMD_SET(1.0f / combined_scale); }

#pragma omp parallel for
    for (size_t i = 0; i < rounded_size; i += SIMD_WIDTH * span) {
        AVX_Data grad_4[span];
        reflow_simd_load<span>(grad_4, grads_half + i);
        if (unscale_grads) { simd_mul<span>(grad_4, grad_4, unscale_factor); }
        AVX_Data momentum_4[span];
        reflow_simd_load<span>(momentum_4, _exp_avg + i);
        AVX_Data variance_4[span];
        reflow_simd_load<span>(variance_4, _exp_avg_sq + i);
        AVX_Data param_4[span];
        reflow_simd_load<span>(param_4, _params_fp32 + i);

        if (decay_grad) { simd_fma<span>(grad_4, param_4, weight_decay4, grad_4); }
        simd_mul<span>(momentum_4, momentum_4, betta1_4);
        simd_fma<span>(momentum_4, grad_4, betta1_minus1_4, momentum_4);
        simd_mul<span>(variance_4, variance_4, betta2_4);
        simd_mul<span>(grad_4, grad_4, grad_4);
        simd_fma<span>(variance_4, grad_4, betta2_minus1_4, variance_4);
        simd_sqrt<span>(grad_4, variance_4);
        simd_fma<span>(grad_4, grad_4, bias2_sqrt, eps_4);
        simd_div<span>(grad_4, momentum_4, grad_4);
        if (decay_param) { simd_fma<span>(param_4, param_4, weight_decay4, param_4); }
        simd_fma<span>(param_4, grad_4, step_size_4, param_4);

        if (store_half_params) {
            // Param generation leaves the FP32 master and the state to the later commit.
            reflow_simd_store_stream<span>(half_params + i, param_4);
        } else {
            reflow_simd_store<span>(_params_fp32 + i, param_4);
            reflow_simd_store<span>(_exp_avg + i, momentum_4);
            reflow_simd_store<span>(_exp_avg_sq + i, variance_4);
        }
    }
}

#endif

int reflow_create_adam_optimizer(int optimizer_id,
                                 float alpha = 1e-3,
                                 float betta1 = 0.9,
                                 float betta2 = 0.999,
                                 float eps = 1e-8,
                                 float weight_decay = 0,
                                 bool adamw_mode = true,
                                 bool should_log = false,
                                 int num_threads = -1);

int reflow_ds_adam_step_params_halfgrad(int optimizer_id,
                                        size_t step,
                                        float lr,
                                        float beta1,
                                        float beta2,
                                        float epsilon,
                                        float weight_decay,
                                        bool bias_correction,
                                        torch::Tensor& params_fp32,
                                        torch::Tensor& grads,
                                        torch::Tensor& exp_avg,
                                        torch::Tensor& exp_avg_sq,
                                        torch::Tensor& half_params,
                                        float combined_scale = 1.0f,
                                        bool skip_increment_step = false);

int reflow_ds_adam_state_step_halfgrad(int optimizer_id,
                                       size_t step,
                                       float lr,
                                       float beta1,
                                       float beta2,
                                       float epsilon,
                                       float weight_decay,
                                       bool bias_correction,
                                       torch::Tensor& params_fp32,
                                       torch::Tensor& grads,
                                       torch::Tensor& exp_avg,
                                       torch::Tensor& exp_avg_sq,
                                       float combined_scale = 1.0f,
                                       bool skip_increment_step = false);

int reflow_destroy_adam_optimizer(int optimizer_id);

/* BF16/FP16 gradient accumulation using AVX (CPU-side gradient accumulation):
 * dst[i] += src[i]. Returns 0 on success.
 */
int reflow_ds_bf16_accumulate(torch::Tensor& dst, torch::Tensor& src);
