// Copyright (c) Microsoft Corporation.
// SPDX-License-Identifier: Apache-2.0

// DeepSpeed Team

#include <c10/cuda/CUDAStream.h>
#include "mixed_gemm.h"
#include "mixed_gemm_api.h"
#include "weight_variant.h"

// Switch helpers inspired by
// https://github.com/NVIDIA/DALI/blob/main/include/dali/core/static_switch.h
// and https://github.com/pytorch/pytorch/blob/master/aten/src/ATen/Dispatch.h

#define ACT_DTYPE_SWITCH(COND, ...)                \
    [&] {                                          \
        if (COND) {                                \
            using ActivationDtype = __half;        \
            return __VA_ARGS__();                  \
        } else {                                   \
            using ActivationDtype = __nv_bfloat16; \
            return __VA_ARGS__();                  \
        }                                          \
    }()

#define WEIGHT_VARIANT_SWITCH(COND, ...)                            \
    [&] {                                                           \
        if (COND) {                                                 \
            constexpr WeightVariant WVariant = WeightVariant::kFP8; \
            return __VA_ARGS__();                                   \
        } else {                                                    \
            constexpr WeightVariant WVariant = WeightVariant::kFP4; \
            return __VA_ARGS__();                                   \
        }                                                           \
    }()

void mixed_gemm(at::Tensor& output,
                at::Tensor& hidden_states,
                at::Tensor& weight,
                at::Tensor& scales,
                c10::optional<at::Tensor>& bias,
                int num_bits,
                int activation_raw)
{
    TORCH_CHECK(output.dtype() == hidden_states.dtype(),
                "Output and hidden states must have the same dtype");
    TORCH_CHECK(num_bits == 4 || num_bits == 8, "Data width must be 4 or 8");
    TORCH_CHECK(output.size(0) == hidden_states.size(0), "Token dimension mismatch");

    int32_t m = output.size(0);
    int32_t k = hidden_states.size(1);
    int32_t n = weight.size(1);

    TORCH_CHECK(weight.size(0) == k, "Weight dimension mismatch");

    ACT_DTYPE_SWITCH(hidden_states.dtype() == torch::kFloat16, [&] {
        WEIGHT_VARIANT_SWITCH(num_bits == 8, [&] {
            fastertransformer::CutlassFpAIntBGemmRunner<ActivationDtype, WVariant> runner =
                *MixedGemmContext<ActivationDtype, WVariant>::Instance().GeMM_Runner();

            ActivationType activation_type = (ActivationType)activation_raw;
            if (!bias.has_value() && activation_type == ActivationType::IDENTITY) {
                runner.gemm((ActivationDtype*)hidden_states.data_ptr(),
                            (const char*)weight.data_ptr(),
                            (ActivationDtype*)scales.data_ptr(),
                            (ActivationDtype*)output.data_ptr(),
                            m,
                            n,
                            k,
                            nullptr,
                            0,
                            at::cuda::getCurrentCUDAStream());
                return;
            } else {
                ActivationDtype* bias_ptr = nullptr;
                if (bias.has_value()) { bias_ptr = (ActivationDtype*)bias.value().data_ptr(); }
                runner.gemm_bias_act((ActivationDtype*)hidden_states.data_ptr(),
                                     (char*)weight.data_ptr(),
                                     (ActivationDtype*)scales.data_ptr(),
                                     bias_ptr,
                                     (ActivationDtype*)output.data_ptr(),
                                     m,
                                     n,
                                     k,
                                     activation_type,
                                     nullptr,
                                     0,
                                     at::cuda::getCurrentCUDAStream());
                return;
            }
        });
    });
}
