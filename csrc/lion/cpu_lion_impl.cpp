// Copyright (c) Microsoft Corporation.
// SPDX-License-Identifier: Apache-2.0

// DeepSpeed Team

#include <torch/extension.h>
#include <cassert>
#include <cmath>
#include <functional>
#include <iostream>
#include <map>
#include <memory>
#include <type_traits>
#include <unordered_map>
#include "cpu_lion.h"

using namespace std::string_literals;
static std::unordered_map<int, std::shared_ptr<void>> s_optimizers;

// C++ interface

template <typename ds_params_precision_t, typename ds_state_precision_t>
void Lion_Optimizer::Step_1(ds_params_precision_t* _params,
                            ds_params_precision_t* grads,
                            ds_state_precision_t* _exp_avg,
                            size_t _param_size)
{
    size_t rounded_size = 0;
#if defined(__AVX512__) or defined(__AVX256__)
    Step_AVX<1>(&rounded_size, _params, grads, _exp_avg, _param_size);
#endif
    if (_param_size > rounded_size) {
        float betta1_minus1 = 1 - _betta1;
        float betta2_minus1 = 1 - _betta2;

        float alpha = _alpha;
        float after_decay = 1 - alpha * _weight_decay;

        for (size_t t = rounded_size; t < _param_size; t += TILE) {
            size_t copy_size = TILE;
            if ((t + TILE) > _param_size) copy_size = _param_size - t;
            size_t offset = copy_size + t;
#pragma omp parallel for
            for (size_t k = t; k < offset; k++) {
                float grad = (float)grads[k];
                float param = (float)_params[k];
                float momentum = _exp_avg[k];
                float tmp = momentum * _betta1;
                tmp = grad * betta1_minus1 + tmp;
                // Rely on portable C++ methods to manipulate the sign bit of a floating-point
                // number.
                tmp = -std::copysignf(alpha, tmp);
                if (_weight_decay > 0) {
                    param = param * after_decay + tmp;
                } else {
                    param = param + tmp;
                }
                momentum = momentum * _betta2;
                momentum = grad * betta2_minus1 + momentum;
                _params[k] = param;
                _exp_avg[k] = momentum;
            }
        }
    }
}

template <typename ds_params_precision_t, typename ds_state_precision_t>
void Lion_Optimizer::Step_4(ds_params_precision_t* _params,
                            ds_params_precision_t* grads,
                            ds_state_precision_t* _exp_avg,
                            size_t _param_size)
{
    size_t rounded_size = 0;
#if defined(__AVX512__) or defined(__AVX256__)
    Step_AVX<4>(&rounded_size, _params, grads, _exp_avg, _param_size);
#endif
    if (_param_size > rounded_size)
        Step_1((_params + rounded_size),
               (grads + rounded_size),
               (_exp_avg + rounded_size),
               (_param_size - rounded_size));
}

int create_lion_optimizer(int optimizer_id,
                          float alpha,
                          float betta1,
                          float betta2,
                          float weight_decay,
                          bool should_log)
{
    auto opt = std::make_shared<Lion_Optimizer>(alpha, betta1, betta2, weight_decay);

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

        printf("Lion Optimizer #%d is created with %s arithmetic capability.\n",
               optimizer_id,
               avx_type.c_str());
        printf("Config: alpha=%f, betas=(%f, %f), weight_decay=%f\n",
               alpha,
               betta1,
               betta2,
               weight_decay);
    }

    return 0;
}

template <typename ds_params_precision_t, typename ds_state_precision_t>
void Lion_Optimizer::Step_8(ds_params_precision_t* _params,
                            ds_params_precision_t* grads,
                            ds_state_precision_t* _exp_avg,
                            size_t _param_size)
{
    size_t rounded_size = 0;
#if defined(__AVX512__) or defined(__AVX256__)
    Step_AVX<8>(&rounded_size, _params, grads, _exp_avg, _param_size);
#endif
    if (_param_size > rounded_size)
        Step_4((_params + rounded_size),
               (grads + rounded_size),
               (_exp_avg + rounded_size),
               (_param_size - rounded_size));
}

template <typename ds_params_precision_t, typename ds_state_precision_t>
void step_invoker(std::shared_ptr<Lion_Optimizer> opt,
                  void* _params,
                  void* grads,
                  void* _exp_avg,
                  size_t _param_size)
{
    opt->Step_8((ds_params_precision_t*)(_params),
                (ds_params_precision_t*)(grads),
                (ds_state_precision_t*)(_exp_avg),
                _param_size);
}

std::map<std::tuple<c10::ScalarType, c10::ScalarType>,
         std::function<void(std::shared_ptr<Lion_Optimizer>, void*, void*, void*, size_t)>>
    invokers;

// Fill map with template functions for each type
template <class ds_params_precision_t, class ds_state_precision_t>
void create_invoker()
{
    invokers[std::tuple(c10::CppTypeToScalarType<ds_params_precision_t>(),
                        c10::CppTypeToScalarType<ds_state_precision_t>())] =
        step_invoker<ds_params_precision_t, ds_state_precision_t>;
}
struct InvokerInitializer {
    InvokerInitializer()
    {
        create_invoker<c10::Half, float>();
        create_invoker<c10::Half, c10::Half>();
        create_invoker<c10::BFloat16, float>();
        create_invoker<c10::BFloat16, c10::BFloat16>();
        create_invoker<float, float>();
    }
} _invoker_initializer;

void invoke(std::shared_ptr<Lion_Optimizer> opt,
            torch::Tensor& params,
            torch::Tensor& grads,
            torch::Tensor& exp_avg,
            size_t param_size)
{
    c10::ScalarType params_type = at::typeMetaToScalarType(params.options().dtype());
    c10::ScalarType state_type = at::typeMetaToScalarType(exp_avg.options().dtype());

    auto it = invokers.find(std::tuple(params_type, state_type));
    if (it == invokers.end()) {
        throw std::runtime_error("Lion optimizer with param type "s + c10::toString(params_type) +
                                 " and state type "s + c10::toString(state_type) +
                                 " is not supported on current hardware"s);
    }

    it->second(opt, params.data_ptr(), grads.data_ptr(), exp_avg.data_ptr(), param_size);
}

int ds_lion_step(int optimizer_id,
                 size_t step,
                 float lr,
                 float beta1,
                 float beta2,
                 float weight_decay,
                 torch::Tensor& params,
                 torch::Tensor& grads,
                 torch::Tensor& exp_avg)
{
    auto params_c = params.contiguous();
    auto grads_c = grads.contiguous();
    auto exp_avg_c = exp_avg.contiguous();

    std::shared_ptr<Lion_Optimizer> opt =
        std::static_pointer_cast<Lion_Optimizer>(s_optimizers[optimizer_id]);
    opt->IncrementStep(step, beta1, beta2);
    opt->update_state(lr, weight_decay);

    invoke(opt, params_c, grads_c, exp_avg_c, params_c.numel());

    return 0;
}

int destroy_lion_optimizer(int optimizer_id)
{
    s_optimizers.erase(optimizer_id);

    return 0;
}
