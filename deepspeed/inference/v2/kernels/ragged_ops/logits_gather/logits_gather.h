// Copyright (c) Microsoft Corporation.
// SPDX-License-Identifier: Apache-2.0

// DeepSpeed Team

#pragma once

#include <c10/cuda/CUDAStream.h>
#include <torch/extension.h>
#include "logits_gather.cuh"
#include "ragged_dtypes.h"

/*
Logits gather will parse the ragged batch data structure and gather only the logits that
will be used for token sampling.
*/
void gather_for_logits(torch::Tensor& final_token_acts,
                       torch::Tensor& all_acts,
                       torch::Tensor& batch_metadata,
                       torch::Tensor& seq_metadata);
