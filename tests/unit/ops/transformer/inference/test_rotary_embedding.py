# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
"""InferenceContext.get_rotary has to build a rotary embedding the installed transformers accepts.

transformers changed `LlamaRotaryEmbedding` twice. Up to 4.37 its forward takes a token
count; from 4.38 it takes `position_ids`. Up to 4.44 the constructor is `(dim, base=..., device=...)`,
4.45 to 4.47 accept either that or a config, and from 4.48 only a config. The fallback attention
path in `softmax_context.py` used the oldest shape of both, so on a current release it raised
`TypeError: __init__() got an unexpected keyword argument 'base'` before reaching any kernel.

No accelerator needed: this covers the construction and the rope values, and the rest of the
fallback path is unchanged.
"""

import inspect

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


def _takes_position_ids(rotary):
    return "position_ids" in inspect.signature(rotary.forward).parameters


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

    x = torch.zeros(1, 1, seq_len, rotary_dim)
    if _takes_position_ids(rotary):
        cos, sin = rotary(x, torch.arange(seq_len).unsqueeze(0))
        assert cos.shape == (1, seq_len, rotary_dim)
        cos, sin = cos[0], sin[0]
    else:
        cos, sin = rotary(x, seq_len)
        assert cos.shape == (seq_len, rotary_dim)

    expected_cos, expected_sin = _reference_cos_sin(rotary_dim, rope_theta, seq_len)
    torch.testing.assert_close(cos, expected_cos)
    torch.testing.assert_close(sin, expected_sin)


def test_get_rotary_uses_rotary_dim_not_the_config_default(context):
    """rotary_dim has to reach the embedding, not the LlamaConfig hidden_size // heads default."""
    rotary = context.get_rotary(32, 10000.0)

    assert rotary.inv_freq.numel() == 16


def test_get_rotary_is_cached(context):
    first = context.get_rotary(64, 10000.0)

    assert context.get_rotary(64, 10000.0) is first


class _StopAfterRotary(Exception):
    """Ends the fallback at the call under test, before it wants a workspace."""


def _run_fallback_to_the_rotary_block(monkeypatch, context):
    """Run `softmax_context_fallback` up to `apply_rotary_pos_emb` and return what it was called with."""
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

    return recorded, position_ids


def test_a_pre_4_38_rotary_gets_a_token_count_and_the_position_ids(monkeypatch, context):
    """transformers <= 4.37: forward(x, seq_len) returns tables that apply_rotary_pos_emb indexes.

    Uses a stand-in with that signature so it runs the same on every installed release.
    """
    seen = {}

    class LegacyRotary:

        def forward(self, x, seq_len=None):
            seen["seq_len"] = seq_len
            return torch.zeros(seq_len, 16), torch.zeros(seq_len, 16)

        __call__ = forward

    context.rotary = LegacyRotary()
    monkeypatch.setattr(context, "max_out_tokens", 1024)

    recorded, position_ids = _run_fallback_to_the_rotary_block(monkeypatch, context)

    assert seen["seq_len"] == 1024
    assert len(recorded["args"]) == 5
    assert recorded["args"][4] is position_ids
    assert not recorded["kwargs"]


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
    if not _takes_position_ids(context.get_rotary(16, 10000.0)):
        pytest.skip("this transformers release predates position_ids in the rotary forward")
    recorded, _ = _run_fallback_to_the_rotary_block(monkeypatch, context)

    assert len(recorded["args"]) == 4, \
        f"the fallback passed {len(recorded['args'])} positional arguments; the fifth is unsqueeze_dim"
    assert not recorded["kwargs"], f"unexpected keyword arguments: {sorted(recorded['kwargs'])}"
