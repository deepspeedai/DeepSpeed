// Copyright (c) Microsoft Corporation.
// SPDX-License-Identifier: Apache-2.0

// DeepSpeed Team

#pragma once

#include <c10/cuda/CUDAStream.h>
#include <torch/extension.h>
#include "activation_type.h"

template <typename T>
void launch_bias_activation(T* activation,
                            const T* bias,
                            const int32_t n_rows,
                            const int32_t n_cols,
                            const ActivationType activation_type,
                            cudaStream_t stream);

void bias_activation(torch::Tensor& activation,
                     c10::optional<torch::Tensor>& bias,
                     const int32_t activation_type);
