// Copyright (c) Microsoft Corporation.
// SPDX-License-Identifier: Apache-2.0

// DeepSpeed Team

#include "cpu_adagrad.h"
#include <torch/extension.h>
#include <functional>
#include <iostream>
#include <map>
#include <memory>
#include <type_traits>
#include <unordered_map>

using namespace std::string_literals;
static std::unordered_map<int, std::shared_ptr<void>> s_optimizers;

// C++ interface

template <typename ds_params_precision_t, typename ds_state_precision_t>
void Adagrad_Optimizer::Step_1(ds_params_precision_t* _params,
                               ds_params_precision_t* grads,
                               ds_state_precision_t* _exp_avg_sq,
                               size_t _param_size)
{
    size_t rounded_size = 0;
#if defined(__AVX512__) or defined(__AVX256__)
    Step_AVX<1>(&rounded_size, _params, grads, _exp_avg_sq, _param_size);
#endif
    if (_param_size > rounded_size) {
        float step_size = -1 * _alpha;
        for (size_t t = rounded_size; t < _param_size; t += TILE) {
            size_t copy_size = TILE;
            if ((t + TILE) > _param_size) copy_size = _param_size - t;
            size_t offset = copy_size + t;
#pragma omp parallel for
            for (size_t k = t; k < offset; k++) {
                float grad = (float)grads[k];
                float param = (float)_params[k];
                float momentum = grads[k];
                float variance = _exp_avg_sq[k];
                if (_weight_decay > 0) { grad = param * _weight_decay + grad; }

                variance += grad * grad;

                grad = sqrt(variance);
                grad += _eps;
                grad = momentum / grad;
                param = grad * step_size + param;
                _params[k] = param;
                // STORE UPDATE TERM TO GRAD'S MEMORY
                grads[k] = grad * step_size;
                _exp_avg_sq[k] = variance;
            }
        }
    }
}

template <typename ds_params_precision_t, typename ds_state_precision_t>
void Adagrad_Optimizer::Step_4(ds_params_precision_t* _params,
                               ds_params_precision_t* grads,
                               ds_state_precision_t* _exp_avg_sq,
                               size_t _param_size)
{
    size_t rounded_size = 0;
#if defined(__AVX512__) or defined(__AVX256__)
    Step_AVX<4>(&rounded_size, _params, grads, _exp_avg_sq, _param_size);
#endif
    if (_param_size > rounded_size)
        Step_1((_params + rounded_size),
               (grads + rounded_size),
               (_exp_avg_sq + rounded_size),
               (_param_size - rounded_size));
}

int create_adagrad_optimizer(int optimizer_id,
                             float alpha = 1e-2,
                             float eps = 1e-8,
                             float weight_decay = 0,
                             bool should_log = false)
{
    auto opt = std::make_shared<Adagrad_Optimizer>(alpha, eps, weight_decay);

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

        printf("Adagrad Optimizer #%d is created with %s arithmetic capability.\n",
               optimizer_id,
               avx_type.c_str());
        printf("Config: alpha=%f, weight_decay=%f\n", alpha, weight_decay);
    }

    return 0;
}

template <typename ds_params_precision_t, typename ds_state_precision_t>
void Adagrad_Optimizer::Step_8(ds_params_precision_t* _params,
                               ds_params_precision_t* grads,
                               ds_state_precision_t* _exp_avg_sq,
                               size_t _param_size)
{
    size_t rounded_size = 0;
#if defined(__AVX512__) or defined(__AVX256__)
    Step_AVX<8>(&rounded_size, _params, grads, _exp_avg_sq, _param_size);
#endif
    if (_param_size > rounded_size)
        Step_4((_params + rounded_size),
               (grads + rounded_size),
               (_exp_avg_sq + rounded_size),
               (_param_size - rounded_size));
}

template <typename ds_params_precision_t, typename ds_state_precision_t>
void step_invoker(std::shared_ptr<Adagrad_Optimizer> opt,
                  void* _params,
                  void* grads,
                  void* _exp_avg_sq,
                  size_t _param_size)
{
    opt->Step_8((ds_params_precision_t*)(_params),
                (ds_params_precision_t*)(grads),
                (ds_state_precision_t*)(_exp_avg_sq),
                _param_size);
}

std::map<std::tuple<c10::ScalarType, c10::ScalarType>,
         std::function<void(std::shared_ptr<Adagrad_Optimizer>, void*, void*, void*, size_t)>>
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

void invoke(std::shared_ptr<Adagrad_Optimizer> opt,
            torch::Tensor& params,
            torch::Tensor& grads,
            torch::Tensor& exp_avg_sq,
            size_t param_size)
{
    c10::ScalarType params_type = at::typeMetaToScalarType(params.options().dtype());
    c10::ScalarType state_type = at::typeMetaToScalarType(exp_avg_sq.options().dtype());

    auto it = invokers.find(std::tuple(params_type, state_type));
    if (it == invokers.end()) {
        throw std::runtime_error("Adagrad optimizer with param type "s +
                                 c10::toString(params_type) + " and state type "s +
                                 c10::toString(state_type) +
                                 " is not supported on current hardware"s);
    }

    it->second(opt, params.data_ptr(), grads.data_ptr(), exp_avg_sq.data_ptr(), param_size);
}

int ds_adagrad_step(int optimizer_id,
                    size_t step,
                    float lr,
                    float epsilon,
                    float weight_decay,
                    torch::Tensor& params,
                    torch::Tensor& grads,
                    torch::Tensor& exp_avg_sq)
{
    auto params_c = params.contiguous();
    auto grads_c = grads.contiguous();
    auto exp_avg_sq_c = exp_avg_sq.contiguous();

    std::shared_ptr<Adagrad_Optimizer> opt =
        std::static_pointer_cast<Adagrad_Optimizer>(s_optimizers[optimizer_id]);
    opt->IncrementStep(step);
    opt->update_state(lr, epsilon, weight_decay);

    invoke(opt, params_c, grads_c, exp_avg_sq_c, params_c.numel());

    return 0;
}

int destroy_adagrad_optimizer(int optimizer_id)
{
    s_optimizers.erase(optimizer_id);

    return 0;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("adagrad_update", &ds_adagrad_step, "DeepSpeed CPU Adagrad update (C++)");
    m.def("create_adagrad", &create_adagrad_optimizer, "DeepSpeed CPU Adagrad (C++)");
    m.def("destroy_adagrad", &destroy_adagrad_optimizer, "DeepSpeed CPU Adagrad destroy (C++)");
}
