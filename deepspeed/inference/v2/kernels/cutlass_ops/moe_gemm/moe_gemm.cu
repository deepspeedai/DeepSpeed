// Copyright (c) Microsoft Corporation.
// SPDX-License-Identifier: Apache-2.0

// DeepSpeed Team

#include <c10/cuda/CUDAStream.h>
#include "moe_gemm.h"
#include "moe_gemm_api.h"
#include "weight_variant.h"

// Switch helpers inspired by
// https://github.com/NVIDIA/DALI/blob/main/include/dali/core/static_switch.h
// and https://github.com/pytorch/pytorch/blob/master/aten/src/ATen/Dispatch.h

#define HIDDEN_DTYPE_SWITCH(COND, ...)                               \
    [&] {                                                            \
        if (COND) {                                                  \
            using ActivationDtype = __half;                          \
            constexpr WeightVariant WVariant = WeightVariant::kFP16; \
            return __VA_ARGS__();                                    \
        } else {                                                     \
            using ActivationDtype = __nv_bfloat16;                   \
            constexpr WeightVariant WVariant = WeightVariant::kBF16; \
            return __VA_ARGS__();                                    \
        }                                                            \
    }()

void moe_gemm(at::Tensor& output,
              at::Tensor& hidden_states,
              at::Tensor& weight,
              c10::optional<at::Tensor>& bias,
              at::Tensor& total_rows_before_expert,
              int activation_raw)
{
    TORCH_CHECK(output.dtype() == hidden_states.dtype(),
                "Output and hidden states must have the same dtype");
    TORCH_CHECK(output.dtype() == weight.dtype(), "Output and weight must have the same dtype");

    int64_t total_rows = hidden_states.size(0);
    int64_t gemm_k = hidden_states.size(1);
    int64_t gemm_n = weight.size(2);
    int num_experts = weight.size(0);

    TORCH_CHECK(total_rows == output.size(0), "Total rows dimension mismatch");
    TORCH_CHECK(gemm_k == weight.size(1), "GEMM K dimension mismatch");
    TORCH_CHECK(gemm_n == output.size(1), "GEMM N dimension mismatch");
    TORCH_CHECK(num_experts == total_rows_before_expert.size(0), "Number of experts mismatch");

    HIDDEN_DTYPE_SWITCH(hidden_states.dtype() == torch::kFloat16, [&] {
        fastertransformer::MoeGemmRunner<ActivationDtype, WVariant> runner =
            *MoeGemmContext<ActivationDtype, WVariant>::Instance().GeMM_Runner();

        ActivationType activation_type = (ActivationType)activation_raw;
        if (!bias.has_value() && activation_type == ActivationType::IDENTITY) {
            runner.moe_gemm((ActivationDtype*)hidden_states.data_ptr(),
                            (char*)weight.data_ptr(),
                            nullptr,
                            (ActivationDtype*)output.data_ptr(),
                            (int64_t*)total_rows_before_expert.data_ptr(),
                            total_rows,
                            gemm_n,
                            gemm_k,
                            num_experts,
                            at::cuda::getCurrentCUDAStream());
            return;
        } else {
            ActivationDtype* bias_ptr = nullptr;
            if (bias.has_value()) {
                bias_ptr = (ActivationDtype*)bias.value().data_ptr();
                TORCH_CHECK(num_experts == bias.value().size(0), "Number of experts mismatch");
                TORCH_CHECK(gemm_n == bias.value().size(1), "GEMM N dimension mismatch");
            }
            runner.moe_gemm_bias_act((ActivationDtype*)hidden_states.data_ptr(),
                                     (char*)weight.data_ptr(),
                                     nullptr,
                                     bias_ptr,
                                     (ActivationDtype*)output.data_ptr(),
                                     (int64_t*)total_rows_before_expert.data_ptr(),
                                     total_rows,
                                     gemm_n,
                                     gemm_k,
                                     num_experts,
                                     activation_type,
                                     at::cuda::getCurrentCUDAStream());
            return;
        }
    });
}

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

void mixed_moe_gemm(at::Tensor& output,
                    at::Tensor& hidden_states,
                    at::Tensor& weight,
                    at::Tensor& scales,
                    c10::optional<at::Tensor>& bias,
                    at::Tensor& total_rows_before_expert,
                    int num_bits,
                    int activation_raw)
{
    TORCH_CHECK(output.dtype() == hidden_states.dtype(),
                "Output and hidden states must have the same dtype");

    int64_t total_rows = hidden_states.size(0);
    int64_t gemm_k = hidden_states.size(1);
    int64_t gemm_n = weight.size(2);
    int num_experts = weight.size(0);

    TORCH_CHECK(total_rows == output.size(0), "Total rows dimension mismatch");
    TORCH_CHECK(gemm_k == weight.size(1), "GEMM K dimension mismatch");
    TORCH_CHECK(gemm_n == output.size(1), "GEMM N dimension mismatch");
    TORCH_CHECK(num_experts == total_rows_before_expert.size(0), "Number of experts mismatch");

    ACT_DTYPE_SWITCH(hidden_states.dtype() == torch::kFloat16, [&] {
        WEIGHT_VARIANT_SWITCH(num_bits == 8, [&] {
            fastertransformer::MoeGemmRunner<ActivationDtype, WVariant> runner =
                *MoeGemmContext<ActivationDtype, WVariant>::Instance().GeMM_Runner();

            ActivationType activation_type = (ActivationType)activation_raw;
            if (!bias.has_value() && activation_type == ActivationType::IDENTITY) {
                runner.moe_gemm((ActivationDtype*)hidden_states.data_ptr(),
                                (char*)weight.data_ptr(),
                                (ActivationDtype*)scales.data_ptr(),
                                (ActivationDtype*)output.data_ptr(),
                                (int64_t*)total_rows_before_expert.data_ptr(),
                                total_rows,
                                gemm_n,
                                gemm_k,
                                num_experts,
                                at::cuda::getCurrentCUDAStream());
                return;
            } else {
                ActivationDtype* bias_ptr = nullptr;
                if (bias.has_value()) {
                    bias_ptr = (ActivationDtype*)bias.value().data_ptr();
                    TORCH_CHECK(num_experts == bias.value().size(0), "Number of experts mismatch");
                    TORCH_CHECK(gemm_n == bias.value().size(1), "GEMM N dimension mismatch");
                }
                runner.moe_gemm_bias_act((ActivationDtype*)hidden_states.data_ptr(),
                                         (char*)weight.data_ptr(),
                                         (ActivationDtype*)scales.data_ptr(),
                                         bias_ptr,
                                         (ActivationDtype*)output.data_ptr(),
                                         (int64_t*)total_rows_before_expert.data_ptr(),
                                         total_rows,
                                         gemm_n,
                                         gemm_k,
                                         num_experts,
                                         activation_type,
                                         at::cuda::getCurrentCUDAStream());
                return;
            }
        });
    });
}
