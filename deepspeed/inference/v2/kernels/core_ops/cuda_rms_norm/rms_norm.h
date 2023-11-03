// Copyright (c) Microsoft Corporation.
// SPDX-License-Identifier: Apache-2.0

// DeepSpeed Team

#pragma once

#include <c10/cuda/CUDAStream.h>
#include <torch/extension.h>
#include "ds_kernel_utils.h"

template <typename T>
void launch_rms_norm(T* norm_output,
                     T* res_output,
                     const T* vals,
                     const T* residual,
                     const T* gamma,
                     float epsilon,
                     int rows,
                     int elems_per_row,
                     cudaStream_t stream);

void rms_norm(torch::Tensor& norm_output,
              torch::Tensor& norm_input,
              torch::Tensor& gamma,
              float epsilon);

void rms_pre_norm(torch::Tensor& norm_output,
                  torch::Tensor& residual_output,
                  torch::Tensor& norm_input,
                  torch::Tensor& residual_input,
                  torch::Tensor& gamma,
                  float epsilon);
