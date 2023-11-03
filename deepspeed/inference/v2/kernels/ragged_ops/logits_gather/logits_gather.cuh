// Copyright (c) Microsoft Corporation.
// SPDX-License-Identifier: Apache-2.0

// DeepSpeed Team

#pragma once

#include "ds_kernel_utils.h"
#include "ragged_dtypes.h"

#ifdef BF16_AVAILABLE
#include <cuda_bf16.h>
#endif

template <typename T>
void launch_logits_gather(T* final_token_acts,
                          const T* all_acts,
                          const RaggedBatchDescriptor* batch_metadata,
                          const InflightSeqDescriptor* seq_metadata,
                          const int32_t n_seqs,
                          const int32_t embed_dim,
                          cudaStream_t stream);
