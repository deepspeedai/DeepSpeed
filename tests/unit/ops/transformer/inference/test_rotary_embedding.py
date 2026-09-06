# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
"""InferenceContext.get_rotary has to build a rotary embedding the installed transformers accepts.

transformers 4.48 replaced `LlamaRotaryEmbedding(dim, base=..., device=...)` with a
config-taking constructor, and its forward went from a token count to `position_ids`. The
fallback attention path in `softmax_context.py` used both of the old shapes, so it raised
`TypeError: __init__() got an unexpected keyword argument 'base'` before reaching any kernel.

No accelerator needed: this covers the construction and the rope values, and the rest of the
fallback path is unchanged.
"""

import inspect

import pytest
import torch

from deepspeed.ops.transformer.inference.op_binding.workspace import InferenceContext


def _reference_cos_sin(rotary_dim, rope_theta, seq_len):
    """cos/sin straight from the rope definition, independent of transformers."""
    inv_freq = 1.0 / (rope_theta**(torch.arange(0, rotary_dim, 2).float() / rotary_dim))
    freqs = torch.outer(torch.arange(seq_len).float(), inv_freq)
    emb = torch.cat((freqs, freqs), dim=-1)
    return emb.cos(), emb.sin()


@pytest.fixture
def context():
    ctx = InferenceContext.Instance()
    ctx.rotary = None
    yield ctx
    ctx.rotary = None


@pytest.mark.parametrize("rotary_dim", [32, 64, 128])
@pytest.mark.parametrize("rope_theta", [10000.0, 500000.0])
def test_get_rotary_matches_the_rope_definition(context, rotary_dim, rope_theta):
    seq_len = 12
    rotary = context.get_rotary(rotary_dim, rope_theta)

    position_ids = torch.arange(seq_len).unsqueeze(0)
    cos, sin = rotary(torch.zeros(1, 1, seq_len, rotary_dim), position_ids)

    expected_cos, expected_sin = _reference_cos_sin(rotary_dim, rope_theta, seq_len)
    assert cos.shape == (1, seq_len, rotary_dim)
    torch.testing.assert_close(cos[0], expected_cos)
    torch.testing.assert_close(sin[0], expected_sin)


def test_get_rotary_uses_rotary_dim_not_the_config_default(context):
    """rotary_dim has to reach the embedding, not the LlamaConfig hidden_size // heads default."""
    rotary = context.get_rotary(32, 10000.0)

    assert rotary.inv_freq.numel() == 16


def test_get_rotary_is_cached(context):
    first = context.get_rotary(64, 10000.0)

    assert context.get_rotary(64, 10000.0) is first


def test_rotary_is_applied_through_cos_sin_not_a_fifth_argument():
    """The fallback hands `apply_rotary_pos_emb` four arguments, and has to.

    transformers 5.0 dropped the deprecated `position_ids` parameter, so the fifth
    positional slot became `unsqueeze_dim`:

        4.51.3 .. 4.57.0   (q, k, cos, sin, position_ids=None, unsqueeze_dim=1)
        5.0.0  .. 5.16.1   (q, k, cos, sin, unsqueeze_dim=1)

    Passing position_ids there reaches `unsqueeze(dim=...)` as a tensor.
    """
    llama = pytest.importorskip("transformers.models.llama.modeling_llama")
    apply_rotary_pos_emb = llama.apply_rotary_pos_emb

    seq_len, rotary_dim = 8, 16
    q = torch.randn(1, 4, seq_len, rotary_dim)
    k = torch.randn(1, 4, seq_len, rotary_dim)
    cos = torch.randn(1, seq_len, rotary_dim)
    sin = torch.randn(1, seq_len, rotary_dim)
    position_ids = torch.arange(seq_len).unsqueeze(0)

    rotated_q, rotated_k = apply_rotary_pos_emb(q, k, cos, sin)
    assert rotated_q.shape == q.shape
    assert rotated_k.shape == k.shape

    fifth = list(inspect.signature(apply_rotary_pos_emb).parameters)[4]
    if fifth == "unsqueeze_dim":
        with pytest.raises(TypeError):
            apply_rotary_pos_emb(q, k, cos, sin, position_ids)
