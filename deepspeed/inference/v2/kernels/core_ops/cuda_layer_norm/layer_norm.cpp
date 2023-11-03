// Copyright (c) Microsoft Corporation.
// SPDX-License-Identifier: Apache-2.0

// DeepSpeed Team

#include "layer_norm.h"

#define DISPATCH_LAYER_NORM(T_TYPE, C_TYPE)                \
    if (input.options().dtype() == torch::T_TYPE) {        \
        launch_fused_ln((C_TYPE*)output.data_ptr(),        \
                        (const C_TYPE*)input.data_ptr(),   \
                        (const C_TYPE*)gamma.data_ptr(),   \
                        (const C_TYPE*)beta.data_ptr(),    \
                        epsilon,                           \
                        rows,                              \
                        elems_per_row,                     \
                        at::cuda::getCurrentCUDAStream()); \
    }

void ds_layer_norm(at::Tensor& output,
                   at::Tensor& input,
                   at::Tensor& gamma,
                   at::Tensor& beta,
                   float epsilon)
{
    bool ragged_input = input.dim() == 2;

    const int rows = ragged_input ? input.size(0) : input.size(0) * input.size(1);
    const int elems_per_row = ragged_input ? input.size(1) : input.size(2);

    DISPATCH_LAYER_NORM(kFloat, float);
    DISPATCH_LAYER_NORM(kHalf, __half);
#ifdef BF16_AVAILABLE
    DISPATCH_LAYER_NORM(kBFloat16, __nv_bfloat16);
#endif
}

#define DISPATCH_LAYER_NORM_RESIDUAL(T_TYPE, C_TYPE)             \
    if (input.options().dtype() == torch::T_TYPE) {              \
        launch_fused_post_ln((C_TYPE*)output.data_ptr(),         \
                             (const C_TYPE*)input.data_ptr(),    \
                             (const C_TYPE*)residual.data_ptr(), \
                             (const C_TYPE*)gamma.data_ptr(),    \
                             (const C_TYPE*)beta.data_ptr(),     \
                             epsilon,                            \
                             rows,                               \
                             elems_per_row,                      \
                             at::cuda::getCurrentCUDAStream());  \
    }

void ds_post_layer_norm(at::Tensor& output,
                        at::Tensor& input,
                        at::Tensor& residual,
                        at::Tensor& gamma,
                        at::Tensor& beta,
                        float epsilon)
{
    bool ragged_input = input.dim() == 2;

    const int rows = ragged_input ? input.size(0) : input.size(0) * input.size(1);
    const int elems_per_row = ragged_input ? input.size(1) : input.size(2);

    DISPATCH_LAYER_NORM_RESIDUAL(kFloat, float);
    DISPATCH_LAYER_NORM_RESIDUAL(kHalf, __half);
#ifdef BF16_AVAILABLE
    DISPATCH_LAYER_NORM_RESIDUAL(kBFloat16, __nv_bfloat16);
#endif
}

#define DISPATCH_PRE_LAYER_NORM_RESIDUAL(T_TYPE, C_TYPE)        \
    if (input.options().dtype() == torch::T_TYPE) {             \
        launch_fused_pre_ln((C_TYPE*)norm_output.data_ptr(),    \
                            (C_TYPE*)res_output.data_ptr(),     \
                            (const C_TYPE*)input.data_ptr(),    \
                            (const C_TYPE*)residual.data_ptr(), \
                            (const C_TYPE*)gamma.data_ptr(),    \
                            (const C_TYPE*)beta.data_ptr(),     \
                            epsilon,                            \
                            rows,                               \
                            elems_per_row,                      \
                            at::cuda::getCurrentCUDAStream());  \
    }

void ds_pre_layer_norm(at::Tensor& res_output,
                       at::Tensor& norm_output,
                       at::Tensor& input,
                       at::Tensor& residual,
                       at::Tensor& gamma,
                       at::Tensor& beta,
                       float epsilon)
{
    bool ragged_input = input.dim() == 2;

    const int rows = ragged_input ? input.size(0) : input.size(0) * input.size(1);
    const int elems_per_row = ragged_input ? input.size(1) : input.size(2);

    DISPATCH_PRE_LAYER_NORM_RESIDUAL(kFloat, float);
    DISPATCH_PRE_LAYER_NORM_RESIDUAL(kHalf, __half);
#ifdef BF16_AVAILABLE
    DISPATCH_PRE_LAYER_NORM_RESIDUAL(kBFloat16, __nv_bfloat16);
#endif
}
