// Copyright (c) Microsoft Corporation.
// SPDX-License-Identifier: Apache-2.0

// DeepSpeed Team

#include "rms_norm.h"

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

void rms_norm(torch::Tensor& norm_output,
              torch::Tensor& norm_input,
              torch::Tensor& gamma,
              float epsilon)
{
    TORCH_CHECK(norm_output.scalar_type() == norm_input.scalar_type(),
                "norm_output and norm_input should have the same data type");
    TORCH_CHECK(norm_output.scalar_type() == gamma.scalar_type(),
                "norm_output and gamma should have the same data type");

    const int32_t rows = norm_input.size(0);
    const int32_t cols = norm_input.size(1);

    TORCH_CHECK(norm_output.size(0) == rows,
                "norm_output and norm_input should have the same first dimension");
    TORCH_CHECK(norm_output.size(1) == cols,
                "norm_output and norm_input should have the same second dimension");

    DISPATCH_FOR_FLOAT(norm_output.scalar_type(), [&] {
        scalar_t* norm_output_ptr = reinterpret_cast<scalar_t*>(norm_output.data_ptr());
        scalar_t* norm_input_ptr = reinterpret_cast<scalar_t*>(norm_input.data_ptr());
        scalar_t* gamma_ptr = reinterpret_cast<scalar_t*>(gamma.data_ptr());
        scalar_t* null_t = nullptr;

        launch_rms_norm(norm_output_ptr,
                        null_t,
                        norm_input_ptr,
                        null_t,
                        gamma_ptr,
                        epsilon,
                        rows,
                        cols,
                        at::cuda::getCurrentCUDAStream());
    });
}

void rms_pre_norm(torch::Tensor& norm_output,
                  torch::Tensor& residual_output,
                  torch::Tensor& norm_input,
                  torch::Tensor& residual_input,
                  torch::Tensor& gamma,
                  float epsilon)
{
    TORCH_CHECK(norm_output.scalar_type() == norm_input.scalar_type(),
                "norm_output and norm_input should have the same data type");
    TORCH_CHECK(norm_output.scalar_type() == gamma.scalar_type(),
                "norm_output and gamma should have the same data type");

    const int32_t rows = norm_input.size(0);
    const int32_t cols = norm_input.size(1);

    TORCH_CHECK(norm_output.size(0) == rows,
                "norm_output and norm_input should have the same first dimension");
    TORCH_CHECK(norm_output.size(1) == cols,
                "norm_output and norm_input should have the same second dimension");

    TORCH_CHECK(residual_output.size(0) == rows,
                "residual_output and norm_input should have the same first dimension");
    TORCH_CHECK(residual_output.size(1) == cols,
                "residual_output and norm_input should have the same second dimension");

    TORCH_CHECK(residual_input.size(0) == rows,
                "residual_input and norm_input should have the same first dimension");
    TORCH_CHECK(residual_input.size(1) == cols,
                "residual_input and norm_input should have the same second dimension");

    DISPATCH_FOR_FLOAT(norm_output.scalar_type(), [&] {
        scalar_t* norm_output_ptr = reinterpret_cast<scalar_t*>(norm_output.data_ptr());
        scalar_t* residual_output_ptr = reinterpret_cast<scalar_t*>(residual_output.data_ptr());
        const scalar_t* norm_input_ptr = reinterpret_cast<const scalar_t*>(norm_input.data_ptr());
        const scalar_t* residual_input_ptr =
            reinterpret_cast<const scalar_t*>(residual_input.data_ptr());
        const scalar_t* gamma_ptr = reinterpret_cast<const scalar_t*>(gamma.data_ptr());

        launch_rms_norm(norm_output_ptr,
                        residual_output_ptr,
                        norm_input_ptr,
                        residual_input_ptr,
                        gamma_ptr,
                        epsilon,
                        rows,
                        cols,
                        at::cuda::getCurrentCUDAStream());
    });
}
