// Copyright (c) Microsoft Corporation.
// SPDX-License-Identifier: Apache-2.0

// DeepSpeed Team

#pragma once

#include <c10/cuda/CUDAStream.h>
#include <torch/extension.h>
#include "ds_kernel_utils.h"

/*
Kernel launch methods for layer norm variants.
*/

template <typename T>
void launch_fused_ln(T* output,
                     const T* vals,
                     const T* gamma,
                     const T* beta,
                     float epsilon,
                     int rows,
                     int elems_per_row,
                     cudaStream_t stream);

template <typename T>
void launch_fused_post_ln(T* output,
                          const T* vals,
                          const T* residual,
                          const T* gamma,
                          const T* beta,
                          float epsilon,
                          int rows,
                          int elems_per_row,
                          cudaStream_t stream);
template <typename T>
void launch_fused_pre_ln(T* norm_output,
                         T* res_output,
                         const T* vals,
                         const T* residual,
                         const T* gamma,
                         const T* beta,
                         float epsilon,
                         int rows,
                         int elems_per_row,
                         cudaStream_t stream);

void ds_layer_norm(at::Tensor& output,
                   at::Tensor& input,
                   at::Tensor& gamma,
                   at::Tensor& beta,
                   float epsilon);

void ds_post_layer_norm(at::Tensor& output,
                        at::Tensor& input,
                        at::Tensor& residual,
                        at::Tensor& gamma,
                        at::Tensor& beta,
                        float epsilon);

void ds_pre_layer_norm(at::Tensor& res_output,
                       at::Tensor& norm_output,
                       at::Tensor& input,
                       at::Tensor& residual,
                       at::Tensor& gamma,
                       at::Tensor& beta,
                       float epsilon);
