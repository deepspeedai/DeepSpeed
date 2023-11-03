// Copyright (c) Microsoft Corporation.
// SPDX-License-Identifier: Apache-2.0

// DeepSpeed Team

#pragma once

#include <cstdint>
#include "cuda.h"

struct AttentionAtom {
    /*
    The attention atom describes the workload of a particular query. The attention
    kernel will execute each ``AttentionAtom`` for each head of the model.
    */

    // Pointer to a list of KV block indices.
    int32_t* block_idx_list;

    // Index of first token in the ragged batch associated with this atom.
    int32_t q_start_idx;

    // Number of tokens in the ragged batch associated with this atom.
    int32_t q_len;

    // Number of key/value blocks associated with this atom. All but the last are
    // assumed to be fully dense.
    int32_t kv_blocks;

    // Number of tokens in the last key/value block.
    int32_t total_extent;

    // Global index of the first token in the atom. For example, in a prompt continuation
    // in which we have already processed 768 tokens, this would be 768.
    int32_t global_q_idx;

    // Unused
    int32_t unused;
};
