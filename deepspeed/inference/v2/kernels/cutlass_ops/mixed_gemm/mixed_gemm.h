// Copyright (c) Microsoft Corporation.
// SPDX-License-Identifier: Apache-2.0

// DeepSpeed Team

#pragma once

#include <torch/extension.h>

void mixed_gemm(at::Tensor& output,
                at::Tensor& hidden_states,
                at::Tensor& weight,
                at::Tensor& scales,
                c10::optional<at::Tensor>& bias,
                int num_bits,
                int activation_raw);
