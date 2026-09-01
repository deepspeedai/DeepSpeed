# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
"""Which parameters per-head Muon tags, and with how many heads. See #8367.

`set_optimizer_flags` already tags `use_muon` per parameter; head structure rides along the
same way so it does not depend on AutoTP being enabled. CPU-only.
"""

from types import SimpleNamespace

import pytest
import torch

import deepspeed
from deepspeed.runtime.config import MUON_OPTIMIZER


class _Attn(torch.nn.Module):

    def __init__(self, hidden=64, q_heads=8, kv_heads=2, head_dim=8, fused=False):
        super().__init__()
        self.q_proj = torch.nn.Linear(hidden, q_heads * head_dim, bias=False)
        self.k_proj = torch.nn.Linear(hidden, kv_heads * head_dim, bias=False)
        self.v_proj = torch.nn.Linear(hidden, kv_heads * head_dim, bias=False)
        self.o_proj = torch.nn.Linear(q_heads * head_dim, hidden, bias=False)
        self.mlp = torch.nn.Linear(hidden, hidden, bias=False)
        self.embed_tokens = torch.nn.Embedding(16, hidden)
        if fused:
            self.qkv_proj = torch.nn.Linear(hidden, (q_heads + 2 * kv_heads) * head_dim, bias=False)
        self.config = SimpleNamespace(num_attention_heads=q_heads, num_key_value_heads=kv_heads)


def _flags(model, per_head=True):
    cfg = SimpleNamespace(optimizer_name=MUON_OPTIMIZER,
                          optimizer_params={"per_head_muon": per_head} if per_head else {})
    deepspeed.set_optimizer_flags(cfg, model)
    return {name: getattr(p, "muon_num_heads", "MISSING") for name, p in model.named_parameters()}


def test_query_and_output_projections_use_the_query_head_count():
    tags = _flags(_Attn(q_heads=8, kv_heads=2))

    assert tags["q_proj.weight"] == 8
    assert tags["o_proj.weight"] == 8


def test_kv_projections_use_the_kv_head_count_under_gqa():
    """K/V have fewer heads than Q under GQA, and splitting them by the query count would be wrong."""
    tags = _flags(_Attn(q_heads=8, kv_heads=2))

    assert tags["k_proj.weight"] == 2
    assert tags["v_proj.weight"] == 2


def test_non_attention_parameters_are_left_on_the_full_matrix_path():
    tags = _flags(_Attn())

    assert tags["mlp.weight"] is None
    assert tags["embed_tokens.weight"] is None


def test_fused_qkv_is_skipped():
    """One matrix holding Q, K and V does not split into uniform heads under GQA."""
    tags = _flags(_Attn(fused=True))

    assert tags["qkv_proj.weight"] is None


def test_opt_in_is_required():
    tags = _flags(_Attn(), per_head=False)

    assert all(v is None for v in tags.values()), tags


def test_shape_that_does_not_divide_is_skipped():
    """A projection whose output dim is not a multiple of the head count is not that layout."""
    model = _Attn(q_heads=8, kv_heads=2)
    model.q_proj = torch.nn.Linear(64, 63, bias=False)  # 63 % 8 != 0

    assert _flags(model)["q_proj.weight"] is None


def test_use_muon_tagging_is_unchanged():
    model = _Attn()
    _flags(model)

    assert model.q_proj.weight.use_muon is True
    assert model.embed_tokens.weight.use_muon is False


@pytest.mark.parametrize("q_heads,kv_heads", [(8, 8), (8, 1), (12, 4)])
def test_head_counts_track_the_config(q_heads, kv_heads):
    tags = _flags(_Attn(q_heads=q_heads, kv_heads=kv_heads, hidden=64, head_dim=8))

    assert tags["q_proj.weight"] == q_heads
    assert tags["k_proj.weight"] == kv_heads
