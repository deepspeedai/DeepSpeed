// Copyright (c) Microsoft Corporation.
// SPDX-License-Identifier: Apache-2.0

// DeepSpeed Team

#pragma once

#include <torch/extension.h>

/*
Construct the attention atoms given the ragged metadata for the current batch.
This could largely be done at the Python level, but since we pack the KV ptr
alongside the int32_t metadata, it gets very ugly to handle the mixed-width
data structures (since we're packing them in a single tensor).
*/
int32_t build_atoms(torch::Tensor& atoms_ten,
                    torch::Tensor& batch_metadata,
                    torch::Tensor& seq_metadata,
                    torch::Tensor& kv_ptrs,
                    const int32_t q_block_size,
                    const int32_t kv_block_size);
