// Copyright (c) Microsoft Corporation.
// SPDX-License-Identifier: Apache-2.0

// DeepSpeed Team

#include <torch/extension.h>
#include <algorithm>
#include <cassert>
#include <cstdint>
#include <cstdio>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <type_traits>
#include <unordered_map>
#include "reflow_cpu_adam.h"

#ifdef __linux__
#include <sched.h>
#include <unistd.h>
#endif

using namespace std::string_literals;

// Independent step counters for param vs. async-state updates.
struct ReflowOptimizerPair {
    std::shared_ptr<Reflow_Adam_Optimizer> param_opt;
    std::shared_ptr<Reflow_Adam_Optimizer> state_opt;
    // Bucket workers share the optimizer scalars; keep their updates serialized.
    std::shared_ptr<std::mutex> param_opt_mutex;
};
static std::unordered_map<int, ReflowOptimizerPair> s_reflow_optimizers;

// Affinity-aware available-core count (falls back to omp_get_num_procs()).
static unsigned int reflow_get_available_cpu_count()
{
#ifdef __linux__
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);

    if (sched_getaffinity(0, sizeof(cpu_set_t), &cpuset) == 0) {
        unsigned int count = 0;
        for (int i = 0; i < CPU_SETSIZE; i++) {
            if (CPU_ISSET(i, &cpuset)) { count++; }
        }
        if (count > 0) { return count; }
    }
#endif
    return omp_get_num_procs();
}

// Non-positive thread counts use all cores available to this worker.
static unsigned int reflow_resolve_omp_threads(int num_threads, unsigned int available)
{
    if (num_threads > 0) { return std::min(static_cast<unsigned int>(num_threads), available); }
    return available;
}

int reflow_create_adam_optimizer(int optimizer_id,
                                 float alpha,
                                 float betta1,
                                 float betta2,
                                 float eps,
                                 float weight_decay,
                                 bool adamw_mode,
                                 bool should_log,
                                 int num_threads)
{
    // The Reflow half-grad kernels exist only for AVX2/AVX-512; fail here instead of at the first
    // step.
#if !(defined(__AVX512__) or defined(__AVX256__))
    throw std::runtime_error("Reflow Adam requires a CPU build with AVX2 or AVX-512 support");
#endif
    auto parameter_opt = std::make_shared<Reflow_Adam_Optimizer>(
        alpha, betta1, betta2, eps, weight_decay, adamw_mode);
    auto state_opt = std::make_shared<Reflow_Adam_Optimizer>(
        alpha, betta1, betta2, eps, weight_decay, adamw_mode);
    parameter_opt->set_num_threads(num_threads);
    state_opt->set_num_threads(num_threads);
    s_reflow_optimizers[optimizer_id] =
        ReflowOptimizerPair{parameter_opt, state_opt, std::make_shared<std::mutex>()};

    if (should_log) {
        std::string avx_type = "";
#if defined(__AVX512__)
        avx_type = "AVX512";
#else
        avx_type = "AVX2";
#endif

        printf("Reflow Adam Optimizer #%d is created with %s arithmetic capability.\n",
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

// Half/BFloat16 gradient + FP16/BF16-store path (params only, skips state).
template <typename ds_store_precision_t,
          typename ds_grad_precision_t,
          typename ds_state_precision_t>
void Reflow_Adam_Optimizer::Step_8_FP16_HalfGrad(float* _params_fp32,
                                                 ds_grad_precision_t* grads_half,
                                                 ds_state_precision_t* _exp_avg,
                                                 ds_state_precision_t* _exp_avg_sq,
                                                 size_t _param_size,
                                                 ds_store_precision_t* half_params,
                                                 float combined_scale)
{
    size_t rounded_size = 0;
#if defined(__AVX512__) or defined(__AVX256__)
    Step_AVX_HalfGrad<8, ds_grad_precision_t, ds_state_precision_t, ds_store_precision_t, true>(
        &rounded_size,
        _params_fp32,
        grads_half,
        _exp_avg,
        _exp_avg_sq,
        _param_size,
        half_params,
        combined_scale);
    // Narrow the SIMD span before handling the remaining scalar tail.
    if (_param_size > rounded_size) {
        size_t tail_rounded_size = 0;
        Step_AVX_HalfGrad<4, ds_grad_precision_t, ds_state_precision_t, ds_store_precision_t, true>(
            &tail_rounded_size,
            _params_fp32 + rounded_size,
            grads_half + rounded_size,
            _exp_avg + rounded_size,
            _exp_avg_sq + rounded_size,
            _param_size - rounded_size,
            half_params + rounded_size,
            combined_scale);
        rounded_size += tail_rounded_size;
    }
    if (_param_size > rounded_size) {
        size_t tail_rounded_size = 0;
        Step_AVX_HalfGrad<1, ds_grad_precision_t, ds_state_precision_t, ds_store_precision_t, true>(
            &tail_rounded_size,
            _params_fp32 + rounded_size,
            grads_half + rounded_size,
            _exp_avg + rounded_size,
            _exp_avg_sq + rounded_size,
            _param_size - rounded_size,
            half_params + rounded_size,
            combined_scale);
        rounded_size += tail_rounded_size;
    }
#endif
    if (_param_size > rounded_size) {
        float betta1_minus1 = 1 - _betta1;
        float betta2_minus1 = 1 - _betta2;

        float step_size = -1 * _alpha / _bias_correction1;
        float w_decay = -1 * _alpha * _weight_decay;
        float unscale_factor = 1.0f / combined_scale;

        for (size_t t = rounded_size; t < _param_size; t += TILE) {
            size_t copy_size = TILE;
            if ((t + TILE) > _param_size) copy_size = _param_size - t;
            size_t offset = copy_size + t;
#pragma omp parallel for
            for (size_t k = t; k < offset; k++) {
                float grad = static_cast<float>(grads_half[k]) * unscale_factor;
                float param = _params_fp32[k];
                float momentum = static_cast<float>(_exp_avg[k]);
                float variance = static_cast<float>(_exp_avg_sq[k]);

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

                // Store param to FP16/BF16 buffer only; state is skipped.
                half_params[k] = ds_store_precision_t(param);
            }
        }
    }
}

// Half/BFloat16 gradient state-only update path.
template <typename ds_grad_precision_t, typename ds_state_precision_t>
void Reflow_Adam_Optimizer::Step_8_State_HalfGrad(float* _params_fp32,
                                                  ds_grad_precision_t* grads_half,
                                                  ds_state_precision_t* _exp_avg,
                                                  ds_state_precision_t* _exp_avg_sq,
                                                  size_t _param_size,
                                                  float combined_scale)
{
    size_t rounded_size = 0;
#if defined(__AVX512__) or defined(__AVX256__)
    Step_AVX_HalfGrad<8, ds_grad_precision_t, ds_state_precision_t, float, false>(
        &rounded_size,
        _params_fp32,
        grads_half,
        _exp_avg,
        _exp_avg_sq,
        _param_size,
        static_cast<float*>(nullptr),
        combined_scale);
    // Narrow the SIMD span before handling the remaining scalar tail.
    if (_param_size > rounded_size) {
        size_t tail_rounded_size = 0;
        Step_AVX_HalfGrad<4, ds_grad_precision_t, ds_state_precision_t, float, false>(
            &tail_rounded_size,
            _params_fp32 + rounded_size,
            grads_half + rounded_size,
            _exp_avg + rounded_size,
            _exp_avg_sq + rounded_size,
            _param_size - rounded_size,
            static_cast<float*>(nullptr),
            combined_scale);
        rounded_size += tail_rounded_size;
    }
    if (_param_size > rounded_size) {
        size_t tail_rounded_size = 0;
        Step_AVX_HalfGrad<1, ds_grad_precision_t, ds_state_precision_t, float, false>(
            &tail_rounded_size,
            _params_fp32 + rounded_size,
            grads_half + rounded_size,
            _exp_avg + rounded_size,
            _exp_avg_sq + rounded_size,
            _param_size - rounded_size,
            static_cast<float*>(nullptr),
            combined_scale);
        rounded_size += tail_rounded_size;
    }
#endif
    if (_param_size > rounded_size) {
        float betta1_minus1 = 1 - _betta1;
        float betta2_minus1 = 1 - _betta2;
        float unscale_factor = 1.0f / combined_scale;
        float step_size = -1.0f * _alpha / _bias_correction1;
        float w_decay = -1.0f * _alpha * _weight_decay;

        for (size_t t = rounded_size; t < _param_size; t += TILE) {
            size_t copy_size = TILE;
            if ((t + TILE) > _param_size) copy_size = _param_size - t;
            size_t offset = copy_size + t;
#pragma omp parallel for
            for (size_t k = t; k < offset; k++) {
                float grad = static_cast<float>(grads_half[k]) * unscale_factor;
                float param = _params_fp32[k];
                float momentum = static_cast<float>(_exp_avg[k]);
                float variance = static_cast<float>(_exp_avg_sq[k]);

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

                _params_fp32[k] = param;
                _exp_avg[k] = static_cast<ds_state_precision_t>(momentum);
                _exp_avg_sq[k] = static_cast<ds_state_precision_t>(variance);
            }
        }
    }
}

/* Half/BFloat16 gradient FP16-store path (params only, state skipped).
 * params_fp32 stays FP32; the half_params buffer receives FP16/BF16.
 */
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
                                        float combined_scale,
                                        bool skip_increment_step)
{
    auto params_c = params_fp32.contiguous();
    auto grads_c = grads.contiguous();
    auto exp_avg_c = exp_avg.contiguous();
    auto exp_avg_sq_c = exp_avg_sq.contiguous();
    auto half_params_c = half_params.contiguous();

    const size_t param_size = params_c.numel();
    if (param_size == 0) { return 0; }

    c10::ScalarType param_type = at::typeMetaToScalarType(params_c.options().dtype());
    c10::ScalarType grad_type = at::typeMetaToScalarType(grads_c.options().dtype());
    c10::ScalarType state_type = at::typeMetaToScalarType(exp_avg_c.options().dtype());
    c10::ScalarType store_type = at::typeMetaToScalarType(half_params_c.options().dtype());

    if (param_type != torch::kFloat32) {
        throw std::runtime_error("reflow_ds_adam_step_params_halfgrad expects FP32 params tensor");
    }
    if (grad_type != torch::kFloat16 && grad_type != torch::kBFloat16) {
        throw std::runtime_error(
            "reflow_ds_adam_step_params_halfgrad expects FP16 or BF16 gradient tensor");
    }
    if (store_type != torch::kFloat16 && store_type != torch::kBFloat16) {
        throw std::runtime_error(
            "reflow_ds_adam_step_params_halfgrad expects FP16 or BF16 half_params tensor");
    }

    ReflowOptimizerPair& pair = s_reflow_optimizers[optimizer_id];
    std::shared_ptr<Reflow_Adam_Optimizer> opt = pair.param_opt;
    // Every worker initializes the shared step scalars before using them; the first range may
    // arrive later. Repeating this initialization for the same step is idempotent.
    // Keep the mutex alive while the worker uses it, even if its registry entry is replaced.
    std::shared_ptr<std::mutex> opt_mutex = pair.param_opt_mutex;
    {
        std::lock_guard<std::mutex> opt_lock(*opt_mutex);
        opt->IncrementStep(step, beta1, beta2);
        opt->update_state(lr, epsilon, weight_decay, bias_correction);
    }
    // Honor the worker's CPU affinity to avoid oversubscribing its pinned cores.
    unsigned int reflow_total_cpus = reflow_get_available_cpu_count();
    unsigned int reflow_max_omp_threads =
        reflow_resolve_omp_threads(opt->get_num_threads(), reflow_total_cpus);
    int reflow_original_omp_threads = omp_get_max_threads();
    omp_set_num_threads(reflow_max_omp_threads);
#if defined(__AVX512__) or defined(__AVX256__)
    if (store_type == torch::kFloat16 && grad_type == torch::kFloat16 &&
        state_type == torch::kFloat32) {
        opt->Step_8_FP16_HalfGrad<c10::Half, c10::Half, float>(params_c.data_ptr<float>(),
                                                               grads_c.data_ptr<c10::Half>(),
                                                               exp_avg_c.data_ptr<float>(),
                                                               exp_avg_sq_c.data_ptr<float>(),
                                                               param_size,
                                                               half_params_c.data_ptr<c10::Half>(),
                                                               combined_scale);
    } else if (store_type == torch::kFloat16 && grad_type == torch::kFloat16 &&
               state_type == torch::kFloat16) {
        opt->Step_8_FP16_HalfGrad<c10::Half, c10::Half, c10::Half>(
            params_c.data_ptr<float>(),
            grads_c.data_ptr<c10::Half>(),
            exp_avg_c.data_ptr<c10::Half>(),
            exp_avg_sq_c.data_ptr<c10::Half>(),
            param_size,
            half_params_c.data_ptr<c10::Half>(),
            combined_scale);
#if defined(__AVX512__) or defined(__AVX256__)
    } else if (store_type == torch::kBFloat16 && grad_type == torch::kBFloat16 &&
               state_type == torch::kFloat32) {
        opt->Step_8_FP16_HalfGrad<c10::BFloat16, c10::BFloat16, float>(
            params_c.data_ptr<float>(),
            grads_c.data_ptr<c10::BFloat16>(),
            exp_avg_c.data_ptr<float>(),
            exp_avg_sq_c.data_ptr<float>(),
            param_size,
            half_params_c.data_ptr<c10::BFloat16>(),
            combined_scale);
    } else if (store_type == torch::kBFloat16 && grad_type == torch::kBFloat16 &&
               state_type == torch::kBFloat16) {
        opt->Step_8_FP16_HalfGrad<c10::BFloat16, c10::BFloat16, c10::BFloat16>(
            params_c.data_ptr<float>(),
            grads_c.data_ptr<c10::BFloat16>(),
            exp_avg_c.data_ptr<c10::BFloat16>(),
            exp_avg_sq_c.data_ptr<c10::BFloat16>(),
            param_size,
            half_params_c.data_ptr<c10::BFloat16>(),
            combined_scale);
#endif
    } else
#endif
    {
        omp_set_num_threads(reflow_original_omp_threads);
        throw std::runtime_error(
            "reflow_ds_adam_step_params_halfgrad: unsupported combination of store/grad/state "
            "dtypes");
    }
    omp_set_num_threads(reflow_original_omp_threads);
    return 0;
}

// Commit the FP32 master and optimizer state from half-precision gradients.
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
                                       float combined_scale,
                                       bool skip_increment_step)
{
    unsigned int total_cpus = reflow_get_available_cpu_count();
    int configured_num_threads = s_reflow_optimizers[optimizer_id].state_opt->get_num_threads();
    unsigned int max_omp_threads = reflow_resolve_omp_threads(configured_num_threads, total_cpus);
    int original_num_threads = omp_get_max_threads();
    omp_set_num_threads(max_omp_threads);

    auto params_c = params_fp32.contiguous();
    auto grads_c = grads.contiguous();
    auto exp_avg_c = exp_avg.contiguous();
    auto exp_avg_sq_c = exp_avg_sq.contiguous();

    const size_t param_size = params_c.numel();
    if (param_size == 0) {
        omp_set_num_threads(original_num_threads);
        return 0;
    }

    c10::ScalarType param_type = at::typeMetaToScalarType(params_c.options().dtype());
    c10::ScalarType grad_type = at::typeMetaToScalarType(grads_c.options().dtype());
    c10::ScalarType state_type = at::typeMetaToScalarType(exp_avg_c.options().dtype());

    if (param_type != torch::kFloat32) {
        omp_set_num_threads(original_num_threads);
        throw std::runtime_error("reflow_ds_adam_state_step_halfgrad expects FP32 params tensor");
    }
    if (grad_type != torch::kFloat16 && grad_type != torch::kBFloat16) {
        omp_set_num_threads(original_num_threads);
        throw std::runtime_error(
            "reflow_ds_adam_state_step_halfgrad expects FP16 or BF16 gradient tensor");
    }

    std::shared_ptr<Reflow_Adam_Optimizer> opt = s_reflow_optimizers[optimizer_id].state_opt;
    if (!skip_increment_step) {
        opt->IncrementStep(step, beta1, beta2);
        opt->update_state(lr, epsilon, weight_decay, bias_correction);
    }
#if defined(__AVX512__) or defined(__AVX256__)
    if (grad_type == torch::kFloat16 && state_type == torch::kFloat32) {
        opt->Step_8_State_HalfGrad<c10::Half, float>(params_c.data_ptr<float>(),
                                                     grads_c.data_ptr<c10::Half>(),
                                                     exp_avg_c.data_ptr<float>(),
                                                     exp_avg_sq_c.data_ptr<float>(),
                                                     param_size,
                                                     combined_scale);
    } else if (grad_type == torch::kFloat16 && state_type == torch::kFloat16) {
        opt->Step_8_State_HalfGrad<c10::Half, c10::Half>(params_c.data_ptr<float>(),
                                                         grads_c.data_ptr<c10::Half>(),
                                                         exp_avg_c.data_ptr<c10::Half>(),
                                                         exp_avg_sq_c.data_ptr<c10::Half>(),
                                                         param_size,
                                                         combined_scale);
#if defined(__AVX512__) or defined(__AVX256__)
    } else if (grad_type == torch::kBFloat16 && state_type == torch::kFloat32) {
        opt->Step_8_State_HalfGrad<c10::BFloat16, float>(params_c.data_ptr<float>(),
                                                         grads_c.data_ptr<c10::BFloat16>(),
                                                         exp_avg_c.data_ptr<float>(),
                                                         exp_avg_sq_c.data_ptr<float>(),
                                                         param_size,
                                                         combined_scale);
    } else if (grad_type == torch::kBFloat16 && state_type == torch::kBFloat16) {
        opt->Step_8_State_HalfGrad<c10::BFloat16, c10::BFloat16>(
            params_c.data_ptr<float>(),
            grads_c.data_ptr<c10::BFloat16>(),
            exp_avg_c.data_ptr<c10::BFloat16>(),
            exp_avg_sq_c.data_ptr<c10::BFloat16>(),
            param_size,
            combined_scale);
#endif
    } else
#endif
    {
        omp_set_num_threads(original_num_threads);
        throw std::runtime_error(
            "reflow_ds_adam_state_step_halfgrad: unsupported combination of grad/state dtypes");
    }
    omp_set_num_threads(original_num_threads);
    return 0;
}

int reflow_destroy_adam_optimizer(int optimizer_id)
{
    s_reflow_optimizers.erase(optimizer_id);

    return 0;
}

/* BF16/FP16 gradient accumulation using AVX: dst[i] += src[i].
 * Both tensors must be contiguous BF16/FP16 tensors on CPU.
 */
int reflow_ds_bf16_accumulate(torch::Tensor& dst, torch::Tensor& src)
{
    auto dst_c = dst.contiguous();
    auto src_c = src.contiguous();

    const size_t param_size = dst_c.numel();
    if (param_size == 0) { return 0; }

    if (src_c.numel() != static_cast<long>(param_size)) {
        throw std::runtime_error("reflow_ds_bf16_accumulate: src and dst must have same numel");
    }

    c10::ScalarType dst_type = at::typeMetaToScalarType(dst_c.options().dtype());
    c10::ScalarType src_type = at::typeMetaToScalarType(src_c.options().dtype());

    if (dst_type != src_type) {
        throw std::runtime_error("reflow_ds_bf16_accumulate: dst and src must have same dtype");
    }

#if defined(__AVX512__) or defined(__AVX256__)
    if (dst_type == torch::kBFloat16) {
        c10::BFloat16* dst_ptr = dst_c.data_ptr<c10::BFloat16>();
        c10::BFloat16* src_ptr = src_c.data_ptr<c10::BFloat16>();

        // AVX512 BF16 accumulation: load as FP32, add, store as BF16.
        constexpr int span = 8;
        size_t rounded_size = ROUND_DOWN(param_size, SIMD_WIDTH * span);

#pragma omp parallel for
        for (size_t i = 0; i < rounded_size; i += SIMD_WIDTH * span) {
            AVX_Data dst_4[span];
            AVX_Data src_4[span];
            reflow_simd_load<span>(dst_4, dst_ptr + i);
            reflow_simd_load<span>(src_4, src_ptr + i);
            simd_add<span>(dst_4, dst_4, src_4);
            reflow_simd_store<span>(dst_ptr + i, dst_4);
        }

        for (size_t i = rounded_size; i < param_size; ++i) {
            float dst_val = static_cast<float>(dst_ptr[i]);
            float src_val = static_cast<float>(src_ptr[i]);
            dst_ptr[i] = static_cast<c10::BFloat16>(dst_val + src_val);
        }
        return 0;
    }
#endif

#if defined(__AVX512__) or defined(__AVX256__)
    if (dst_type == torch::kFloat16) {
        c10::Half* dst_ptr = dst_c.data_ptr<c10::Half>();
        c10::Half* src_ptr = src_c.data_ptr<c10::Half>();

        // AVX FP16 accumulation: load as FP32, add, store as FP16.
        constexpr int span = 8;
        size_t rounded_size = ROUND_DOWN(param_size, SIMD_WIDTH * span);

#pragma omp parallel for
        for (size_t i = 0; i < rounded_size; i += SIMD_WIDTH * span) {
            AVX_Data dst_4[span];
            AVX_Data src_4[span];
            reflow_simd_load<span>(dst_4, dst_ptr + i);
            reflow_simd_load<span>(src_4, src_ptr + i);
            simd_add<span>(dst_4, dst_4, src_4);
            reflow_simd_store<span>(dst_ptr + i, dst_4);
        }

        for (size_t i = rounded_size; i < param_size; ++i) {
            float dst_val = static_cast<float>(dst_ptr[i]);
            float src_val = static_cast<float>(src_ptr[i]);
            dst_ptr[i] = static_cast<c10::Half>(dst_val + src_val);
        }
        return 0;
    }
#endif

    throw std::runtime_error(
        "reflow_ds_bf16_accumulate: only BF16 and FP16 (with AVX2/AVX512) are supported");
}
