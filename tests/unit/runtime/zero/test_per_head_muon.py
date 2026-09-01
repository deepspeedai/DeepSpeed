# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
"""Per-head Muon: Newton-Schulz on each attention head instead of the whole projection.

Full-matrix orthogonalization treats every head as one coupled block, so heads with larger
momentum dominate the shared update direction. Kimi K3 (arXiv:2607.24653 §2.5) and GLM-5
"Muon Split" (arXiv:2602.15763) both orthogonalize per head instead. See #8367.

CPU-only: these pin the arithmetic, not the accelerator path.
"""

import pytest
import torch

from deepspeed.accelerator import get_accelerator
from deepspeed.runtime.zero.muon.original_muon import (
    muon_update,
    zeropower_via_gram_newtonschulz,
    zeropower_via_newtonschulz5,
)


def _ns_tolerance(ns_method):
    """A few ulps of whatever dtype the kernel iterates in.

    `gram` runs in fp16 and `newtonschulz5` in bf16 (fp32 where unsupported), and the iteration
    amplifies rounding, so batched and unbatched NS agree to a handful of ulps rather than
    bitwise. Deriving the bound from the dtype keeps it honest instead of tuned to pass.
    """
    if ns_method == "gram":
        dtype = torch.float16 if get_accelerator().is_fp16_supported() else torch.float32
    else:
        dtype = torch.bfloat16 if get_accelerator().is_bf16_supported() else torch.float32
    eps = torch.finfo(dtype).eps
    return dict(rtol=8 * eps, atol=8 * eps)


def _norm_rtol(ns_method):
    """Scale agreement: a couple of ulps of the compute dtype, and never looser than 1%."""
    if ns_method == "gram":
        dtype = torch.float16 if get_accelerator().is_fp16_supported() else torch.float32
    else:
        dtype = torch.bfloat16 if get_accelerator().is_bf16_supported() else torch.float32
    return max(1e-2, 2 * torch.finfo(dtype).eps)


def _update_only(grad, momentum, beta=0.95, nesterov=True):
    """The pre-orthogonalization update muon_update forms, without mutating the caller's tensors."""
    grad, momentum = grad.clone(), momentum.clone()
    momentum.lerp_(grad, 1 - beta)
    return grad.lerp_(momentum, beta) if nesterov else momentum


@pytest.mark.parametrize("ns_method", ["gram", "newtonschulz5"])
@pytest.mark.parametrize("num_heads,head_dim,in_features", [(4, 8, 32), (2, 16, 32), (8, 4, 64)])
def test_per_head_matches_orthogonalizing_each_head_alone(ns_method, num_heads, head_dim, in_features):
    """The batched path must equal running NS on each head block on its own."""
    torch.manual_seed(0)
    out_features = num_heads * head_dim
    grad = torch.randn(out_features, in_features)
    momentum = torch.randn(out_features, in_features)

    got = muon_update(grad.clone(), momentum.clone(), ns_method=ns_method, num_heads=num_heads)

    update = _update_only(grad, momentum)
    ns_fn = zeropower_via_gram_newtonschulz if ns_method == "gram" else zeropower_via_newtonschulz5
    scale = max(1, head_dim / in_features)**0.5
    expected = torch.cat([ns_fn(update[h * head_dim:(h + 1) * head_dim], steps=5) * scale
                          for h in range(num_heads)]).to(got.dtype)

    assert got.shape == (out_features, in_features)
    torch.testing.assert_close(got, expected, **_ns_tolerance(ns_method))
    # Elementwise agreement is ulp-limited, so also pin the overall scale.
    torch.testing.assert_close(got.norm(), expected.norm(), rtol=_norm_rtol(ns_method), atol=0.0)


@pytest.mark.parametrize("ns_method", ["gram", "newtonschulz5"])
def test_single_head_reproduces_the_full_matrix_path(ns_method):
    """num_heads=1 is the whole projection, so it has to agree with the existing behaviour."""
    torch.manual_seed(0)
    grad = torch.randn(16, 32)
    momentum = torch.randn(16, 32)

    per_head = muon_update(grad.clone(), momentum.clone(), ns_method=ns_method, num_heads=1)
    full = muon_update(grad.clone(), momentum.clone(), ns_method=ns_method)

    # Batched and unbatched NS take the same arithmetic path but not bit-identically in the
    # half-precision compute dtype, so compare at that granularity.
    torch.testing.assert_close(per_head, full, **_ns_tolerance(ns_method))
    torch.testing.assert_close(per_head.norm(), full.norm(), rtol=_norm_rtol(ns_method), atol=0.0)


def test_per_head_differs_from_full_matrix_when_heads_are_unbalanced():
    """The point of the change: one loud head must stop setting the direction for the quiet ones.

    Without this the test would pass even if num_heads were ignored.
    """
    torch.manual_seed(0)
    num_heads, head_dim, in_features = 4, 8, 32
    grad = torch.randn(num_heads * head_dim, in_features)
    grad[:head_dim] *= 100.0  # one head with a far larger gradient scale
    momentum = torch.zeros_like(grad)

    per_head = muon_update(grad.clone(), momentum.clone(), num_heads=num_heads)
    full = muon_update(grad.clone(), momentum.clone())

    quiet = slice(head_dim, None)
    assert not torch.allclose(per_head[quiet], full[quiet], rtol=1e-2, atol=1e-2)
    # Every head should come out with a comparable update scale.
    norms = torch.stack([per_head[h * head_dim:(h + 1) * head_dim].norm() for h in range(num_heads)])
    assert norms.max() / norms.min() < 1.5


def test_rejects_shapes_that_do_not_split_into_heads():
    grad = torch.randn(15, 32)
    momentum = torch.zeros_like(grad)

    with pytest.raises(ValueError, match="not divisible by num_heads"):
        muon_update(grad.clone(), momentum.clone(), num_heads=4)

    conv_like = torch.randn(4, 4, 3, 3)
    with pytest.raises(ValueError, match="expects a 2D attention projection"):
        muon_update(conv_like.clone(), torch.zeros_like(conv_like), num_heads=4)
