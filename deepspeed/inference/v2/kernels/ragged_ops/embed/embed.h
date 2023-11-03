// Copyright (c) Microsoft Corporation.
// SPDX-License-Identifier: Apache-2.0

// DeepSpeed Team

#pragma once

#include <c10/cuda/CUDAStream.h>
#include <torch/extension.h>
#include "embed.cuh"

/*
Embeddings kernel aware of ragged batch structure.
*/
void ragged_embed(torch::Tensor& embedded_tokens,
                  torch::Tensor& input_ids,
                  torch::Tensor& embedding_weight,
                  c10::optional<torch::Tensor>& position_weight,
                  int32_t position_embed_offset,
                  torch::Tensor& batch_metadata,
                  torch::Tensor& seq_metadata,
                  torch::Tensor& tokens_to_seq,
                  torch::Tensor& kv_ptrs);
