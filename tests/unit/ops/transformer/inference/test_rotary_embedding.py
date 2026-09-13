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

from deepspeed.ops.transformer.inference.config import DeepSpeedInferenceConfig
from deepspeed.ops.transformer.inference.op_binding.softmax_context import SoftmaxContextOp
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


class _StopAfterRotary(Exception):
    """Ends the fallback at the call under test, before it wants a workspace."""


def test_the_fallback_passes_apply_rotary_pos_emb_four_arguments(monkeypatch, context):
    """Asserts this repo's call, not the library's signature.

    transformers 5.0 dropped the deprecated `position_ids` parameter, so the fifth
    positional slot became `unsqueeze_dim`:

        4.51.3 .. 4.57.0   (q, k, cos, sin, position_ids=None, unsqueeze_dim=1)
        5.0.0  .. 5.16.1   (q, k, cos, sin, unsqueeze_dim=1)

    Passing `position_ids` there reaches `unsqueeze(dim=...)` as a tensor. Checking the
    library's own signature would not catch that, since the mistake is in the caller; the
    recorded call has to come from `softmax_context_fallback` itself.

    The recorder raises so execution stops at the rotary block. `update_cache` sits two
    lines below and needs a workspace, which is not what this is about.
    """
    llama = pytest.importorskip("transformers.models.llama.modeling_llama")

    recorded = {}

    def recorder(*args, **kwargs):
        recorded["args"], recorded["kwargs"] = args, kwargs
        raise _StopAfterRotary

    monkeypatch.setattr(llama, "apply_rotary_pos_emb", recorder)

    heads, head_dim, seq_len, rotary_dim = 4, 16, 8, 16
    query_key_value = torch.randn(1, seq_len, 3 * heads * head_dim)
    position_ids = torch.arange(seq_len).unsqueeze(0)

    config = DeepSpeedInferenceConfig(hidden_size=heads * head_dim, heads=heads, dtype=torch.float32)
    op = SoftmaxContextOp.__new__(SoftmaxContextOp)
    op.config = config

    with pytest.raises(_StopAfterRotary):
        op.softmax_context_fallback(query_key_value, None, rotary_dim, True, False, heads, heads, 1.0, False, False, 0,
                                    False, 0, 1, None, 10000.0, True, 0, position_ids)

    assert len(recorded["args"]) == 4, \
        f"the fallback passed {len(recorded['args'])} positional arguments; the fifth is unsqueeze_dim"
    assert not recorded["kwargs"], f"unexpected keyword arguments: {sorted(recorded['kwargs'])}"
