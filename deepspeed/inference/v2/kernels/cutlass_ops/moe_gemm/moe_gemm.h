// Copyright (c) Microsoft Corporation.
// SPDX-License-Identifier: Apache-2.0

// DeepSpeed Team

#pragma once

#include <torch/extension.h>

void moe_gemm(at::Tensor& output,
              at::Tensor& hidden_states,
              at::Tensor& weight,
              c10::optional<at::Tensor>& bias,
              at::Tensor& total_rows_before_expert,
              int activation_raw);

void mixed_moe_gemm(at::Tensor& output,
                    at::Tensor& hidden_states,
                    at::Tensor& weight,
                    at::Tensor& scales,
                    c10::optional<at::Tensor>& bias,
                    at::Tensor& total_rows_before_expert,
                    int num_bits,
                    int activation_raw);
