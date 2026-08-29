# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team
"""Compare the fused weighted restore with the eager reference."""

import pytest
import torch

from deepspeed.accelerator import get_accelerator
from deepspeed.module_inject.auto_ep_layer import combine_from_routed
from deepspeed.ops.triton_ops import autoep_fused_token_ops as fused_ops


def _fused_engine_available():
    accelerator = get_accelerator()
    return (accelerator.is_available() and accelerator.device_name().startswith("cuda") and fused_ops.is_available())


pytestmark = pytest.mark.skipif(not _fused_engine_available(),
                                reason="the fused weighted restore needs CUDA and Triton")


def _device():
    return get_accelerator().current_device_name()


@pytest.mark.parametrize("top_k", [2, 4, 6, 8])
@pytest.mark.parametrize("hidden", [128, 130])
@pytest.mark.parametrize("row_dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("score_dtype", [torch.float32, torch.bfloat16])
def test_fused_weighted_restore_matches_eager_including_gradients(top_k, hidden, row_dtype, score_dtype):
    device = _device()
    num_tokens, num_experts = 24, 8
    generator = torch.Generator(device=device).manual_seed(20260824)

    selected_experts = torch.randint(0, num_experts, (num_tokens, top_k), device=device, generator=generator)
    token_indices_sorted = torch.argsort(selected_experts.view(-1), stable=True)
    assert not torch.equal(token_indices_sorted, torch.arange(num_tokens * top_k, device=device))

    rows = torch.randn(num_tokens * top_k, hidden, device=device, dtype=row_dtype, generator=generator)
    scores = torch.rand(num_tokens, top_k, device=device, dtype=score_dtype, generator=generator)
    upstream = torch.randn(1, num_tokens, hidden, device=device, dtype=row_dtype, generator=generator)

    eager_rows = rows.clone().requires_grad_(True)
    eager_scores = scores.clone().requires_grad_(True)
    eager_output = combine_from_routed(
        eager_rows,
        top_scores=eager_scores,
        token_indices_sorted=token_indices_sorted,
        top_k=top_k,
        score_apply="post",
        combine_impl="weighted_sum",
        shape=(1, num_tokens, hidden),
    )

    fused_rows = rows.clone().requires_grad_(True)
    fused_scores = scores.clone().requires_grad_(True)
    fused_output = fused_ops.fused_weighted_restore(
        fused_rows,
        top_scores=fused_scores,
        token_indices_sorted=token_indices_sorted,
        top_k=top_k,
        shape=(1, num_tokens, hidden),
    )

    output_tolerance = {"rtol": 1e-5, "atol": 1e-6} if row_dtype == torch.float32 else {}
    torch.testing.assert_close(fused_output, eager_output, **output_tolerance)

    eager_output.backward(upstream)
    fused_output.backward(upstream)

    torch.testing.assert_close(fused_rows.grad, eager_rows.grad, **output_tolerance)
    # Hidden reduction order only affects the last bits of FP32 score gradients.
    score_tolerance = {"rtol": 1e-4, "atol": 1e-5} if score_dtype == torch.float32 else {}
    torch.testing.assert_close(fused_scores.grad, eager_scores.grad, **score_tolerance)


def test_fused_weighted_restore_supports_double_backward():
    device = _device()
    num_tokens, top_k, hidden = 4, 2, 16
    token_indices_sorted = torch.tensor([2, 0, 7, 1, 4, 6, 3, 5], device=device)
    rows = torch.randn(num_tokens * top_k, hidden, device=device, dtype=torch.bfloat16, requires_grad=True)
    scores = torch.rand(num_tokens, top_k, device=device, dtype=torch.float32, requires_grad=True)

    output = fused_ops.fused_weighted_restore(
        rows,
        top_scores=scores,
        token_indices_sorted=token_indices_sorted,
        top_k=top_k,
        shape=(1, num_tokens, hidden),
    )
    grad_rows, grad_scores = torch.autograd.grad(output.float().sum(), (rows, scores), create_graph=True)
    score_cross_gradient = torch.autograd.grad(grad_rows.float().sum(), scores, retain_graph=True)[0]
    row_cross_gradient = torch.autograd.grad(grad_scores.float().sum(), rows)[0]

    assert torch.isfinite(score_cross_gradient).all()
    assert torch.isfinite(row_cross_gradient).all()
    assert score_cross_gradient.abs().sum() > 0
    assert row_cross_gradient.abs().sum() > 0


def test_fused_engine_names_what_it_cannot_run():
    device = _device()
    for dtype in fused_ops.SUPPORTED_ROW_DTYPES:
        fused_ops.assert_supported(torch.randn(8, 16, device=device, dtype=dtype), score_apply="post")

    with pytest.raises(RuntimeError, match="bfloat16, float16, and float32"):
        fused_ops.assert_supported(torch.randn(8, 16, device=device, dtype=torch.float64), score_apply="post")

    supported = torch.randn(8, 16, device=device, dtype=torch.bfloat16)
    with pytest.raises(RuntimeError, match='resolved score_apply="pre"'):
        fused_ops.assert_supported(supported, score_apply="pre")

    with pytest.raises(RuntimeError, match="CUDA kernels"):
        fused_ops.assert_supported(torch.randn(8, 16, dtype=torch.bfloat16), score_apply="post")
