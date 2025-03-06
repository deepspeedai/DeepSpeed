// Copyright (c) Microsoft Corporation.
// SPDX-License-Identifier: Apache-2.0

// DeepSpeed Team

#include <torch/extension.h>

#include "atom_builder.h"
#include "blocked_flash.h"
#include "blocked_kv_rotary.h"
#include "embed.h"
#include "logits_gather.h"
#include "moe_gather.h"
#include "moe_scatter.h"
#include "top_k_gating.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    // atom_builder.h
    m.def("build_atoms", &build_atoms, "Host kernel for building the atoms.");

    // blocked_flash.h
    m.def("flash_attn_by_atoms",
          &flash_attn_by_atoms,
          "Blocked flash attention scheduled with atoms");

    // blocked_kv_rotary.h
    m.def("kv_rotary_embeddings", &kv_rotary_embeddings, "KV rotary embedding for blocked KV");
    m.def("kv_trained_rotary_embeddings",
          &kv_trained_rotary_embeddings,
          "KV rotary embeddings for blocked KV");
    m.def("linear_kv_copy", &linear_kv_copy, "Linear copy for blocked KV");

    // embed.h
    m.def("ragged_embed", &ragged_embed, "Embedding lookup for ragged batch");

    // logits_gather.h
    m.def("gather_for_logits", &gather_for_logits, "Sparse gather from ragged batch");

    // moe_gather.h
    m.def("moe_gather", &moe_gather, "MoE gather for top-1-gating.");

    // moe_scatter.h
    m.def("moe_scatter", &moe_scatter, "MoE scatter for top-1-gating.");

    // top_k_gating.h
    m.def("top_k_gating", &top_k_gating, "Top-1 gating for MoE with ragged batch awareness.");
}
