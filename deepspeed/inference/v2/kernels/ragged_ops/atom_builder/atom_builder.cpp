// Copyright (c) Microsoft Corporation.
// SPDX-License-Identifier: Apache-2.0

// DeepSpeed Team

#include "atom_builder.h"
#include "attention_atom.h"
#include "ragged_dtypes.h"

int32_t build_atoms(torch::Tensor& atoms_ten,
                    torch::Tensor& batch_metadata,
                    torch::Tensor& seq_metadata,
                    torch::Tensor& kv_ptrs,
                    const int32_t q_block_size,
                    const int32_t kv_block_size)
{
    const RaggedBatchDescriptor* batch_desc =
        reinterpret_cast<const RaggedBatchDescriptor*>(batch_metadata.data_ptr());

    const InflightSeqDescriptor* seq_desc =
        reinterpret_cast<const InflightSeqDescriptor*>(seq_metadata.data_ptr());

    int32_t** kv_ptr_list = reinterpret_cast<int32_t**>(kv_ptrs.data_ptr());

    AttentionAtom* atoms = reinterpret_cast<AttentionAtom*>(atoms_ten.data_ptr());

    int32_t n_atoms = 0;
    for (int i = 0; i < batch_desc->n_sequences; i++) {
        const int seq_atoms = (seq_desc[i].n_tokens + q_block_size - 1) / q_block_size;
        int32_t cur_start_idx = seq_desc[i].start_idx;
        int32_t global_start_idx = seq_desc[i].seen_tokens;
        int32_t remaining_toks = seq_desc[i].n_tokens;

        for (int j = 0; j < seq_atoms; j++) {
            atoms[n_atoms].block_idx_list = kv_ptr_list[i];
            atoms[n_atoms].q_start_idx = cur_start_idx;
            atoms[n_atoms].q_len = std::min(remaining_toks, q_block_size);
            atoms[n_atoms].global_q_idx = global_start_idx;

            const int32_t end_toks = global_start_idx + atoms[n_atoms].q_len;
            // TODO(cmikeh2): This logic needs to be changed for sparse implementations
            atoms[n_atoms].kv_blocks = (end_toks + kv_block_size - 1) / kv_block_size;
            atoms[n_atoms].total_extent = end_toks;

            cur_start_idx += atoms[n_atoms].q_len;
            global_start_idx += atoms[n_atoms].q_len;
            remaining_toks -= atoms[n_atoms].q_len;
            n_atoms++;
        }
    }

    return n_atoms;
}
