// Copyright (c) Microsoft Corporation.
// SPDX-License-Identifier: Apache-2.0

// DeepSpeed Team

#include "logits_gather.h"

#define DISPATCH_TO_LOGITS_GATHER(T_TYPE, C_TYPE)                  \
    if (all_acts.options().dtype() == torch::T_TYPE) {             \
        launch_logits_gather((C_TYPE*)final_token_acts.data_ptr(), \
                             (const C_TYPE*)all_acts.data_ptr(),   \
                             batch_metadata_raw,                   \
                             seq_metadata_raw,                     \
                             n_seqs,                               \
                             embed_dim,                            \
                             at::cuda::getCurrentCUDAStream());    \
    }

/*
Logits gather will parse the ragged batch data structure and gather only the logits that
will be used for token sampling.
*/
void gather_for_logits(torch::Tensor& final_token_acts,
                       torch::Tensor& all_acts,
                       torch::Tensor& batch_metadata,
                       torch::Tensor& seq_metadata)
{
    const RaggedBatchDescriptor* batch_metadata_raw =
        reinterpret_cast<const RaggedBatchDescriptor*>(batch_metadata.data_ptr());

    const InflightSeqDescriptor* seq_metadata_raw =
        reinterpret_cast<const InflightSeqDescriptor*>(seq_metadata.data_ptr());

    const int n_seqs = final_token_acts.size(0);
    const int embed_dim = final_token_acts.size(1);

    TORCH_CHECK(all_acts.scalar_type() == final_token_acts.scalar_type(),
                "all_acts and final_token_acts must have the same scalar type");

    DISPATCH_TO_LOGITS_GATHER(kFloat, float)
    DISPATCH_TO_LOGITS_GATHER(kHalf, half)
#ifdef BF16_AVAILABLE
    DISPATCH_TO_LOGITS_GATHER(kBFloat16, __nv_bfloat16)
#endif
}
