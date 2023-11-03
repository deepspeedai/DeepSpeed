// Copyright (c) Microsoft Corporation.
// SPDX-License-Identifier: Apache-2.0

// DeepSpeed Team

#include "ragged_kernel_helpers.h"

BatchWrapperCPP make_cpp_batch_wrapper(torch::Tensor& batch_metadata,
                                       torch::Tensor& seq_metadata,
                                       torch::Tensor& tokens_to_seq,
                                       torch::Tensor& kv_cache_desc,
                                       int32_t block_size,
                                       int32_t n_blocks)
{
    const RaggedBatchDescriptor* batch_metadata_raw =
        reinterpret_cast<const RaggedBatchDescriptor*>(batch_metadata.data_ptr());

    const InflightSeqDescriptor* seq_metadata_raw =
        reinterpret_cast<const InflightSeqDescriptor*>(seq_metadata.data_ptr());

    const int32_t* tokens_to_seq_raw = tokens_to_seq.data_ptr<int32_t>();

    int32_t** kv_ptrs_raw = reinterpret_cast<int32_t**>(kv_cache_desc.data_ptr());
    KVCacheDescriptor kv_desc = {kv_ptrs_raw, block_size, n_blocks};

    BatchWrapperCPP wrapper = {batch_metadata_raw, seq_metadata_raw, tokens_to_seq_raw, kv_desc};
    return wrapper;
}
