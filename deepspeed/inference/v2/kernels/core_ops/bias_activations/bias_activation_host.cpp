// Copyright (c) Microsoft Corporation.
// SPDX-License-Identifier: Apache-2.0

// DeepSpeed Team

#include "bias_activation.h"
#include <c10/cuda/CUDAStream.h>
#include "ds_kernel_utils.h"

#ifdef BF16_AVAILABLE
#define DTYPE_SWITCH(DTYPE, ...)                                        \
    [&] {                                                               \
        if (DTYPE == torch::kFloat16) {                                 \
            using scalar_t = __half;                                    \
            return __VA_ARGS__();                                       \
        } else if (DTYPE == torch::kBFloat16) {                         \
            using scalar_t = __nv_bfloat16;                             \
            return __VA_ARGS__();                                       \
        } else {                                                        \
            TORCH_CHECK(false, "Unsupported dtype for BiasActivation"); \
        }                                                               \
    }()
#else
#define DTYPE_SWITCH(DTYPE, ...)                                        \
    [&] {                                                               \
        if (DTYPE == torch::kFloat16) {                                 \
            using scalar_t = __half;                                    \
            return __VA_ARGS__();                                       \
        } else {                                                        \
            TORCH_CHECK(false, "Unsupported dtype for BiasActivation"); \
        }                                                               \
    }()
#endif

/*
In-place bias and activation fusion kernel.
*/
void bias_activation(torch::Tensor& activation,
                     c10::optional<torch::Tensor>& bias,
                     const int32_t act_type)
{
    const ActivationType atype = static_cast<ActivationType>(act_type);
    const int32_t rows = activation.size(0);
    const int32_t cols = activation.size(1);

    TORCH_CHECK(atype == ActivationType::GELU || atype == ActivationType::RELU ||
                    atype == ActivationType::SILU || atype == ActivationType::IDENTITY,
                "Unsupported activation type for BiasActivation");
    TORCH_CHECK(activation.dim() == 2, "BiasActivation only supports 2D activation tensors");

    DTYPE_SWITCH(activation.scalar_type(), [&] {
        scalar_t* activation_ptr = reinterpret_cast<scalar_t*>(activation.data_ptr());

        const scalar_t* bias_ptr;
        if (bias.has_value()) {
            TORCH_CHECK(activation.scalar_type() == bias.value().scalar_type(),
                        "BiasActivation activation and bias must have same dtype");
            bias_ptr = reinterpret_cast<const scalar_t*>(bias.value().data_ptr());
        } else {
            bias_ptr = nullptr;
        }

        if (atype == ActivationType::IDENTITY && bias_ptr == nullptr) { return; }

        launch_bias_activation<scalar_t>(
            activation_ptr, bias_ptr, rows, cols, atype, c10::cuda::getCurrentCUDAStream());
    });
}
