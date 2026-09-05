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
