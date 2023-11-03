// Copyright (c) Microsoft Corporation.
// SPDX-License-Identifier: Apache-2.0

// DeepSpeed Team

#pragma once

#include <torch/extension.h>
#include "ragged_dtypes.h"

BatchWrapperCPP make_cpp_batch_wrapper(torch::Tensor& batch_metadata,
                                       torch::Tensor& seq_metadata,
                                       torch::Tensor& tokens_to_seq,
                                       torch::Tensor& kv_cache_desc,
                                       int32_t block_size,
                                       int32_t n_blocks);
