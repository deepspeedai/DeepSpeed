// Copyright (c) Microsoft Corporation.
// SPDX-License-Identifier: Apache-2.0

// DeepSpeed Team

#pragma once

#include "ds_kernel_utils.h"
#include "ragged_dtypes.h"

#ifdef BF16_AVAILABLE
#include <cuda_bf16.h>
#endif

template <typename TokenType, typename EmbedType>
void launch_ragged_embed_kernel(EmbedType* embedded_tokens,
                                const TokenType* input_ids,
                                const EmbedType* embedding_weight,
                                const EmbedType* position_weight,
                                const BatchWrapperCPP batch_desc,
                                const int32_t n_tokens,
                                const int32_t embed_dim,
                                const int32_t vocab_size,
                                const int32_t max_position_embed_idx,
                                const int32_t position_embed_offset,
                                cudaStream_t stream);
