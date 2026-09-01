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
from deepspeed import _attention_head_count
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


def test_query_projection_uses_the_query_head_count():
    tags = _flags(_Attn(q_heads=8, kv_heads=2))

    assert tags["q_proj.weight"] == 8


def test_output_projection_is_left_alone():
    """o_proj is `[hidden, num_heads * head_dim]` - its heads are on the input axis.

    The split is on dim 0, so tagging it would cut across the wrong axis, and with the usual
    hidden == num_heads * head_dim it still divides evenly, i.e. silently wrong rather than an
    error. Regression test: it was tagged in the first version of this.
    """
    tags = _flags(_Attn(q_heads=8, kv_heads=2))

    assert tags["o_proj.weight"] is None


@pytest.mark.parametrize("mlp_name", [
    "intermediate.dense.weight",
    "output.dense.weight",
    "mlp.dense_h_to_4h.weight",
    "mlp.dense_4h_to_h.weight",
])
def test_mlp_matrices_named_dense_are_not_treated_as_attention(mlp_name):
    """`dense` names an MLP matrix as often as an attention one.

    Matching it anywhere in the path tagged `intermediate.dense` and `dense_h_to_4h` with a head
    count, splitting a matrix that has no head structure. Regression test: the first version of
    this matched on the full path and did exactly that.
    """
    from deepspeed import _attention_head_count

    model = _Attn(q_heads=8, kv_heads=2)
    weight = torch.zeros(4 * 64, 64)

    assert _attention_head_count(f"encoder.layer.0.{mlp_name}", weight, model) is None


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


@pytest.mark.parametrize("arch", ["llama", "qwen2", "mistral"])
def test_split_qkv_architectures_tag_only_qkv(arch):
    """Real HF configs rather than a stand-in, so the leaf names are the ones models actually use."""
    transformers = pytest.importorskip("transformers")
    cfg_cls = {
        "llama": transformers.LlamaConfig,
        "qwen2": transformers.Qwen2Config,
        "mistral": transformers.MistralConfig,
    }[arch]
    cfg = cfg_cls(hidden_size=64,
                  num_attention_heads=8,
                  num_key_value_heads=2,
                  num_hidden_layers=1,
                  intermediate_size=128,
                  vocab_size=32)
    model = transformers.AutoModelForCausalLM.from_config(cfg)

    tags = {n.split(".")[-2]: _attention_head_count(n, p, model) for n, p in model.named_parameters() if p.ndim == 2}

    assert tags["q_proj"] == 8
    assert tags["k_proj"] == 2, "GQA: k/v are blocked by num_key_value_heads, not the query count"
    assert tags["v_proj"] == 2
    assert tags["o_proj"] is None, "o_proj's heads are on the input axis"
    for mlp_leaf in ("gate_proj", "up_proj", "down_proj"):
        assert tags[mlp_leaf] is None, f"{mlp_leaf} has no head structure"


@pytest.mark.parametrize("arch", ["gpt_neox", "falcon"])
def test_fused_qkv_architectures_tag_nothing(arch):
    """These name their MLP matrices `dense_h_to_4h` / `dense_4h_to_h` and their output proj `dense`.

    Matching `dense` anywhere in the path tagged all three; this pins that none of them are.
    """
    transformers = pytest.importorskip("transformers")
    cfg_cls = {"gpt_neox": transformers.GPTNeoXConfig, "falcon": transformers.FalconConfig}[arch]
    kwargs = dict(hidden_size=64, num_attention_heads=8, num_hidden_layers=1, vocab_size=32)
    if arch == "gpt_neox":
        kwargs["intermediate_size"] = 128
    model = transformers.AutoModelForCausalLM.from_config(cfg_cls(**kwargs))

    tags = {n: _attention_head_count(n, p, model) for n, p in model.named_parameters() if p.ndim == 2}

    assert all(v is None for v in tags.values()), \
        {k: v for k, v in tags.items() if v is not None}
