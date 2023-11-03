// Copyright (c) Microsoft Corporation.
// SPDX-License-Identifier: Apache-2.0

// DeepSpeed Team

#pragma once

#include <torch/extension.h>

void flash_attn_by_atoms(at::Tensor& out,
                         at::Tensor& q,
                         at::Tensor& k,
                         at::Tensor& v,
                         at::Tensor& attention_atoms,
                         const float softmax_scale,
                         const bool is_causal);
