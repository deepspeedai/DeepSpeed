// Copyright (c) Microsoft Corporation.
// SPDX-License-Identifier: Apache-2.0

// DeepSpeed Team

#include "gated_activation_kernels.h"

#ifdef BF16_AVAILABLE
#define DISPATCH_FOR_FLOAT(DTYPE, ...)                                  \
    [&] {                                                               \
        if (DTYPE == torch::kFloat32) {                                 \
            using scalar_t = float;                                     \
            return __VA_ARGS__();                                       \
        } else if (DTYPE == torch::kFloat16) {                          \
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
#define DISPATCH_FOR_FLOAT(DTYPE, ...)                                  \
    [&] {                                                               \
        if (DTYPE == torch::kFloat32) {                                 \
            using scalar_t = float;                                     \
            return __VA_ARGS__();                                       \
        } else if (DTYPE == torch::kFloat16) {                          \
            using scalar_t = __half;                                    \
            return __VA_ARGS__();                                       \
        } else {                                                        \
            TORCH_CHECK(false, "Unsupported dtype for BiasActivation"); \
        }                                                               \
    }()
#endif

void ds_gated_activation(at::Tensor& output,
                         at::Tensor& input,
                         c10::optional<torch::Tensor>& bias,
                         int activation_type_raw)
{
    bool ragged_input = input.dim() == 2;

    const ActivationType activation_type = static_cast<ActivationType>(activation_type_raw);

    const int rows = ragged_input ? input.size(0) : input.size(0) * input.size(1);
    const int cols = ragged_input ? input.size(1) : input.size(2);

    DISPATCH_FOR_FLOAT(input.scalar_type(), [&] {
        scalar_t* bias_ptr = nullptr;
        if (bias.has_value()) {
            TORCH_CHECK(bias.value().scalar_type() == input.scalar_type(),
                        "Bias type must match input type");
            TORCH_CHECK(bias.value().numel() == cols,
                        "Bias must have the same number of elements as the input channels");
            bias_ptr = reinterpret_cast<scalar_t*>(bias.value().data_ptr());
        }

        scalar_t* output_ptr = reinterpret_cast<scalar_t*>(output.data_ptr());
        const scalar_t* input_ptr = reinterpret_cast<const scalar_t*>(input.data_ptr());

        launch_gated_activation(output_ptr,
                                input_ptr,
                                bias_ptr,
                                rows,
                                cols,
                                activation_type,
                                c10::cuda::getCurrentCUDAStream());
    });
}
