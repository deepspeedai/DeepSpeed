// Copyright (c) Microsoft Corporation.
// SPDX-License-Identifier: Apache-2.0

// DeepSpeed Team

/* Lion keeps a single momentum buffer (exp_avg); there is no second moment, no
 * eps, and no bias correction. The update is:
 *     c_t   = beta1*m_old + (1-beta1)*g          (update direction)
 *     p_new = p*(1 - lr*wd) - lr*sign(c_t)       (decoupled decay + sign step)
 *     m_new = beta2*m_old + (1-beta2)*g          (momentum EMA)
 * Both c_t and m_new read the OLD momentum, so c_t must be formed before the
 * momentum register is overwritten with m_new.
 */

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
#include "reflow_cpu_affinity.h"
#include "reflow_simd.h"

/* Enable madvise(MADV_HUGEPAGE) hints in the streaming kernels (Linux only).
 * Transparent huge pages reduce TLB pressure for the large, sequentially
 * streamed param/grad/state buffers.
 */
#define USE_HUGE_PAGE 1
#ifdef __linux__
#include <sys/mman.h>
#endif

class Reflow_Lion_Optimizer {
public:
    Reflow_Lion_Optimizer(float alpha = 1e-3,
                          float betta1 = 0.9,
                          float betta2 = 0.999,
                          float weight_decay = 0)
        : _alpha(alpha),
          _betta1(betta1),
          _betta2(betta2),
          _weight_decay(weight_decay),
          _step(0),
          _num_threads(-1)
    {
    }
    ~Reflow_Lion_Optimizer() {}

#if defined(__AVX512__) or defined(__AVX256__)
    // Half/BFloat16 gradient path that updates params (FP32 master) + state.
    template <int span, typename ds_grad_precision_t, typename ds_state_precision_t>
    void Step_AVX_HalfGrad(size_t* rounded_size,
                           float* _params_fp32,
                           ds_grad_precision_t* grads_half,
                           ds_state_precision_t* _exp_avg,
                           size_t param_size,
                           float combined_scale = 1.0f);
    // Half/BFloat16 gradient path that updates params only, stored to FP16/BF16.
    template <int span,
              typename ds_store_precision_t,
              typename ds_grad_precision_t,
              typename ds_state_precision_t>
    void Step_AVX_FP16_HalfGrad(size_t* rounded_size,
                                float* _params_fp32,
                                ds_grad_precision_t* grads_half,
                                ds_state_precision_t* _exp_avg,
                                size_t param_size,
                                ds_store_precision_t* half_params,
                                float combined_scale);
#endif

    // Half/BFloat16 gradient wrapper, params-only FP16/BF16 store.
    template <typename ds_store_precision_t,
              typename ds_grad_precision_t,
              typename ds_state_precision_t>
    void Step_8_FP16_HalfGrad(float* _params_fp32,
                              ds_grad_precision_t* grads_half,
                              ds_state_precision_t* _exp_avg,
                              size_t _param_size,
                              ds_store_precision_t* half_params,
                              float combined_scale);

    // Half/BFloat16 gradient wrapper, params (FP32 master) + state update.
    template <typename ds_grad_precision_t, typename ds_state_precision_t>
    void Step_8_State_HalfGrad(float* _params_fp32,
                               ds_grad_precision_t* grads_half,
                               ds_state_precision_t* _exp_avg,
                               size_t _param_size,
                               float combined_scale = 1.0f);

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

    /* -1 (the default) means "use all available cores"; a positive value caps the
     * OpenMP thread count for the CPU-Lion state update.
     */
    inline void set_num_threads(int num_threads) { _num_threads = num_threads; }
    inline int get_num_threads() const { return _num_threads; }

private:
    float _alpha;
    float _betta1;
    float _betta2;
    float _weight_decay;
    size_t _step;

    // OpenMP thread cap for the CPU-Lion state update; -1 means all available cores.
    int _num_threads;
};

#if defined(__AVX512__) or defined(__AVX256__)

/* Half/BFloat16 gradient path: params stay FP32, grads are Half/BFloat16,
 * arithmetic is FP32, and the deferred commit writes the FP32 master and momentum.
 */
template <int span, typename ds_grad_precision_t, typename ds_state_precision_t>
void Reflow_Lion_Optimizer::Step_AVX_HalfGrad(size_t* rounded_size,
                                              float* _params_fp32,
                                              ds_grad_precision_t* grads_half,
                                              ds_state_precision_t* _exp_avg,
                                              size_t _param_size,
                                              float combined_scale)
{
    size_t new_rounded_size = 0;

    AVX_Data neg1_4;
    neg1_4.data = SIMD_SET(-1.0f);

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
    bool use_weight_decay = (_weight_decay > 0);

    // Unscale (after FP32 promotion) only when combined_scale > 1.
    bool need_unscale = (combined_scale > 1.0f);
    AVX_Data unscale_factor;
    if (need_unscale) { unscale_factor.data = SIMD_SET(1.0f / combined_scale); }

    new_rounded_size = ROUND_DOWN(_param_size, SIMD_WIDTH * span);

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
    }
#endif

    ReflowCPUAffinity affinity;
#pragma omp parallel
    {
        affinity.apply();
#pragma omp for schedule(static) nowait
        for (size_t i = 0; i < new_rounded_size; i += SIMD_WIDTH * span) {
            AVX_Data grad_4[span];
            reflow_simd_load<span>(grad_4, grads_half + i);
            if (need_unscale) { simd_mul<span>(grad_4, grad_4, unscale_factor); }
            AVX_Data momentum_4[span];
            reflow_simd_load<span>(momentum_4, _exp_avg + i);
            AVX_Data param_4[span];
            reflow_simd_load<span>(param_4, _params_fp32 + i);

            AVX_Data tmp_4[span];
            simd_mul<span>(tmp_4, momentum_4, betta1_4);
            simd_fma<span>(tmp_4, grad_4, betta1_minus1_4, tmp_4);
            simd_and<span>(tmp_4, tmp_4, neg1_4);
            simd_xor<span>(tmp_4, tmp_4, step_size_4);
            if (use_weight_decay) {
                simd_fma<span>(param_4, param_4, after_decay_4, tmp_4);
            } else {
                simd_add<span>(param_4, param_4, tmp_4);
            }

            simd_mul<span>(momentum_4, momentum_4, betta2_4);
            simd_fma<span>(momentum_4, grad_4, betta2_minus1_4, momentum_4);

            reflow_simd_store<span>(_params_fp32 + i, param_4);
            reflow_simd_store<span>(_exp_avg + i, momentum_4);
        }
    }

    *rounded_size = new_rounded_size;
}

// Step_AVX_FP16_HalfGrad: update params only, store to FP16/BF16 buffer (skips
// the FP32 master and state). This is the foreground Phase-1 BF16 producer.
template <int span,
          typename ds_store_precision_t,
          typename ds_grad_precision_t,
          typename ds_state_precision_t>
void Reflow_Lion_Optimizer::Step_AVX_FP16_HalfGrad(size_t* rounded_size,
                                                   float* _params_fp32,
                                                   ds_grad_precision_t* grads_half,
                                                   ds_state_precision_t* _exp_avg,
                                                   size_t _param_size,
                                                   ds_store_precision_t* half_params,
                                                   float combined_scale)
{
    size_t new_rounded_size = 0;

    AVX_Data neg1_4;
    neg1_4.data = SIMD_SET(-1.0f);

    AVX_Data betta1_4;
    betta1_4.data = SIMD_SET(_betta1);

    float betta1_minus1 = 1 - _betta1;
    AVX_Data betta1_minus1_4;
    betta1_minus1_4.data = SIMD_SET(betta1_minus1);

    float step_size = -_alpha;
    AVX_Data step_size_4;
    step_size_4.data = SIMD_SET(step_size);

    float after_decay = 1.0f - _alpha * _weight_decay;
    AVX_Data after_decay_4;
    if (_weight_decay > 0) after_decay_4.data = SIMD_SET(after_decay);
    bool use_weight_decay = (_weight_decay > 0);

    bool need_unscale = (combined_scale > 1.0f);
    AVX_Data unscale_factor;
    if (need_unscale) { unscale_factor.data = SIMD_SET(1.0f / combined_scale); }

    new_rounded_size = ROUND_DOWN(_param_size, SIMD_WIDTH * span);

#if defined(__linux__) && USE_HUGE_PAGE
    if (new_rounded_size > 0) {
        size_t page_size = 4096;
        auto madvise_bytes = [&](size_t elem_size) {
            return (_param_size * elem_size + page_size - 1) & ~(page_size - 1);
        };
        size_t params_bytes = madvise_bytes(sizeof(float));
        size_t grad_bytes = madvise_bytes(sizeof(ds_grad_precision_t));
        size_t state_bytes = madvise_bytes(sizeof(ds_state_precision_t));
        size_t store_bytes = madvise_bytes(sizeof(ds_store_precision_t));
        void* params_aligned = (void*)((uintptr_t)_params_fp32 & ~(page_size - 1));
        madvise(params_aligned, params_bytes, MADV_HUGEPAGE);
        madvise((void*)((uintptr_t)grads_half & ~(page_size - 1)), grad_bytes, MADV_HUGEPAGE);
        madvise((void*)((uintptr_t)_exp_avg & ~(page_size - 1)), state_bytes, MADV_HUGEPAGE);
        madvise((void*)((uintptr_t)half_params & ~(page_size - 1)), store_bytes, MADV_HUGEPAGE);
    }
#endif

    // Phase-1 reads the OLD momentum but never overwrites it, so the momentum EMA
    // (m_new) is intentionally not computed here -- it is committed in Phase-2.
    ReflowCPUAffinity affinity;
#pragma omp parallel
    {
        affinity.apply();
#pragma omp for schedule(static) nowait
        for (size_t i = 0; i < new_rounded_size; i += SIMD_WIDTH * span) {
            AVX_Data grad_4[span];
            reflow_simd_load<span>(grad_4, grads_half + i);
            if (need_unscale) { simd_mul<span>(grad_4, grad_4, unscale_factor); }
            AVX_Data momentum_4[span];
            reflow_simd_load<span>(momentum_4, _exp_avg + i);
            AVX_Data param_4[span];
            reflow_simd_load<span>(param_4, _params_fp32 + i);

            AVX_Data tmp_4[span];
            simd_mul<span>(tmp_4, momentum_4, betta1_4);
            simd_fma<span>(tmp_4, grad_4, betta1_minus1_4, tmp_4);
            simd_and<span>(tmp_4, tmp_4, neg1_4);
            simd_xor<span>(tmp_4, tmp_4, step_size_4);
            if (use_weight_decay) {
                simd_fma<span>(param_4, param_4, after_decay_4, tmp_4);
            } else {
                simd_add<span>(param_4, param_4, tmp_4);
            }
            reflow_simd_store_stream<span>(half_params + i, param_4);
        }
    }

    *rounded_size = new_rounded_size;
}

#endif

/* Reflow CPU-Lion entry points (own registry, isolated from any base CPU-Lion).
 * The param/state split mirrors the Adam path: the params-halfgrad producer runs
 * during backward and writes only the BF16/FP16 weights; the state-step commit
 * runs on a background worker and writes the FP32 master + exp_avg.
 */
int reflow_create_lion_optimizer(int optimizer_id,
                                 float alpha = 1e-3,
                                 float betta1 = 0.9,
                                 float betta2 = 0.999,
                                 float weight_decay = 0,
                                 bool should_log = false,
                                 int num_threads = -1);

// Half/BFloat16 gradient FP16/BF16-store path (params only, state skipped).
int reflow_ds_lion_step_params_halfgrad(int optimizer_id,
                                        size_t step,
                                        float lr,
                                        float beta1,
                                        float beta2,
                                        float weight_decay,
                                        torch::Tensor& params_fp32,
                                        torch::Tensor& grads,
                                        torch::Tensor& exp_avg,
                                        torch::Tensor& half_params,
                                        float combined_scale,
                                        bool skip_increment_step = false);

// Half/BFloat16 gradient params (FP32 master) + state commit.
int reflow_ds_lion_state_step_halfgrad(int optimizer_id,
                                       size_t step,
                                       float lr,
                                       float beta1,
                                       float beta2,
                                       float weight_decay,
                                       torch::Tensor& params_fp32,
                                       torch::Tensor& grads,
                                       torch::Tensor& exp_avg,
                                       float combined_scale,
                                       bool skip_increment_step = false);

int reflow_destroy_lion_optimizer(int optimizer_id);
