# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import pytest
import torch

from deepspeed.sequence.layer import apply_rotary_pos_emb, _rotate_half


def _make_freqs(seq_len, rot_dim, theta=10000.0, device="cpu"):
    inv_freq = 1.0 / (theta ** (torch.arange(0, rot_dim, 2, device=device).float() / rot_dim))
    t = torch.arange(seq_len, device=device).float()
    freqs = torch.einsum("i,j->ij", t, inv_freq)
    emb = torch.cat((freqs, freqs), dim=-1)
    return emb.cos(), emb.sin()


def _ref_apply_rotary(t, freqs_cos, freqs_sin):
    rot_dim = freqs_cos.shape[-1]
    t, t_pass = t[..., :rot_dim], t[..., rot_dim:]
    t = (t * freqs_cos) + (_rotate_half(t) * freqs_sin)
    return t if t_pass.shape[-1] == 0 else torch.cat((t, t_pass), dim=-1)


@pytest.mark.parametrize("seq_len", [1, 17, 128])
@pytest.mark.parametrize("dim", [32, 64, 128])
@pytest.mark.parametrize("rotary_dim", [None, 16, 32, 64])
def test_apply_rotary_pos_emb(seq_len, dim, rotary_dim):
    rot_dim = rotary_dim if rotary_dim is not None else dim
    if rot_dim > dim or rot_dim % 2 != 0:
        pytest.skip("rotary_dim must be <= dim and even")

    t = torch.randn(seq_len, 4, dim)
    freqs_cos, freqs_sin = _make_freqs(seq_len, rot_dim)
    # unsqueeze a broadcastable heads dim: t is [seq_len, n_heads, dim], freqs is [seq_len, dim]
    freqs_cos = freqs_cos[:, :rot_dim].unsqueeze(1)
    freqs_sin = freqs_sin[:, :rot_dim].unsqueeze(1)

    result = apply_rotary_pos_emb(t, freqs_cos, freqs_sin)
    expected = _ref_apply_rotary(t, freqs_cos, freqs_sin)

    assert torch.allclose(result, expected, atol=1e-6), (
        f"seq_len={seq_len}, dim={dim}, rot_dim={rot_dim}: max diff={((result - expected).abs().max()).item()}"
    )


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_apply_rotary_pos_emb_grad_flow(dtype):
    seq_len, n_heads, dim = 8, 4, 64
    rot_dim = 64
    t = torch.randn(seq_len, n_heads, dim, dtype=dtype, requires_grad=True)
    freqs_cos, freqs_sin = _make_freqs(seq_len, rot_dim)
    freqs_cos = freqs_cos[:, :rot_dim].unsqueeze(1)
    freqs_sin = freqs_sin[:, :rot_dim].unsqueeze(1)

    out = apply_rotary_pos_emb(t, freqs_cos, freqs_sin)
    loss = out.sum()
    loss.backward()

    assert t.grad is not None
    assert not torch.isnan(t.grad).any(), "NaNs in gradient"
    assert t.grad.shape == t.shape, f"grad shape {t.grad.shape} != {t.shape}"
