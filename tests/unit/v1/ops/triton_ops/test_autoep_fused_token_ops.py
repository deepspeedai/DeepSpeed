# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team
"""Compare the fused weighted restore with the eager reference."""

import math

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


def _ordered_ulp_values(tensor):
    raw = tensor.detach().cpu().view(torch.int16).to(torch.int32)
    raw = torch.bitwise_and(raw, 0xffff)
    sign = torch.bitwise_and(raw, 0x8000) != 0
    return torch.where(sign, torch.bitwise_and(torch.bitwise_not(raw), 0xffff), torch.bitwise_or(raw, 0x8000))


def _assert_row_weighting_forward_matches(fused, eager):
    if torch.equal(fused, eager):
        return

    fused_order = _ordered_ulp_values(fused)
    eager_order = _ordered_ulp_values(eager)
    ulp = (fused_order - eager_order).abs()
    max_ulp = int(ulp.max().item()) if ulp.numel() else 0
    mismatch_count = int(torch.count_nonzero(fused.cpu() != eager.cpu()).item())
    raise AssertionError(f"fused row weighting was not bitwise equal to eager; "
                         f"mismatches={mismatch_count}, max_ulp_of_eager_dtype={max_ulp}")


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


@pytest.mark.parametrize("n_rows, hidden", [(37, 2048), (0, 2048), (5, 4096)])
@pytest.mark.parametrize("row_dtype", [torch.bfloat16, torch.float16])
def test_fused_row_weighting_matches_eager_including_gradients(n_rows, hidden, row_dtype):
    device = _device()
    generator = torch.Generator(device=device).manual_seed(20260920 + n_rows + hidden)
    rows = torch.randn(n_rows, hidden, device=device, dtype=row_dtype, generator=generator)
    row_pattern = torch.arange(n_rows, device=device, dtype=torch.float32).reshape(-1, 1)
    weights = torch.randn(n_rows, 1, device=device, dtype=torch.float32, generator=generator)
    weights = weights * 8.0
    if n_rows:
        weights = torch.where(row_pattern % 5 == 0, torch.zeros_like(weights), weights)
        weights = torch.where(row_pattern % 5 == 1, -weights.abs(), weights)
        weights = torch.where(row_pattern % 5 == 2, weights.sign() * 4096.0, weights)
    upstream = torch.randn(n_rows, hidden, device=device, dtype=row_dtype, generator=generator)

    eager_rows = rows.clone().requires_grad_(True)
    eager_weights = weights.clone().requires_grad_(True)
    eager_output = (eager_rows.float() * eager_weights).to(eager_rows.dtype)

    fused_rows = rows.clone().requires_grad_(True)
    fused_weights = weights.clone().requires_grad_(True)
    fused_output = fused_ops.fused_row_weighting(fused_rows, fused_weights)

    assert fused_output.shape == rows.shape
    assert fused_output.dtype == rows.dtype
    assert fused_output.device == rows.device
    assert fused_output.is_contiguous()
    _assert_row_weighting_forward_matches(fused_output, eager_output)

    eager_output.backward(upstream)
    fused_output.backward(upstream)

    assert fused_rows.grad.shape == rows.shape
    assert fused_weights.grad.shape == weights.shape
    _assert_row_weighting_forward_matches(fused_rows.grad, eager_rows.grad)
    torch.testing.assert_close(fused_weights.grad, eager_weights.grad, rtol=1e-3, atol=1e-2)


@pytest.mark.parametrize("n_rows, hidden", [(4096, 2048), (257, 256)])
@pytest.mark.parametrize("row_dtype", [torch.bfloat16, torch.float16])
def test_fused_row_weighting_weight_gradient_is_as_accurate_as_eager(n_rows, hidden, row_dtype):
    """The weight gradient sums the same FP32 products as eager in another order; it must be no less accurate."""
    device = _device()
    generator = torch.Generator(device=device).manual_seed(97 + hidden)
    rows = torch.randn(n_rows, hidden, device=device, generator=generator).to(row_dtype)
    weights = torch.rand(n_rows, 1, device=device, generator=generator)
    upstream = torch.randn(n_rows, hidden, device=device, generator=generator).to(row_dtype)

    eager_weights = weights.clone().requires_grad_(True)
    (rows.float() * eager_weights).to(row_dtype).backward(upstream)
    fused_weights = weights.clone().requires_grad_(True)
    fused_ops.fused_row_weighting(rows, fused_weights).backward(upstream)

    products = upstream.double() * rows.double()
    reference = products.sum(dim=-1, keepdim=True)
    fused_error = (fused_weights.grad.double() - reference).abs()
    eager_error = (eager_weights.grad.double() - reference).abs()
    # A tree of FP32 additions over the hidden dimension, plus the final sum of per-tile partials.
    depth = math.ceil(math.log2(hidden)) + 2
    bound = depth * torch.finfo(torch.float32).eps * products.abs().sum(dim=-1, keepdim=True)
    assert torch.all(fused_error <= bound), f"max error over bound {(fused_error / bound).max().item()}"
    assert fused_error.max() <= 2 * eager_error.max() + 1e-12, (fused_error.max().item(), eager_error.max().item())
    assert fused_error.median() <= 2 * eager_error.median() + 1e-12, (fused_error.median().item(),
                                                                      eager_error.median().item())


def test_fused_row_weighting_guards_name_unsupported_inputs():
    device = _device()
    rows = torch.randn(8, 16, device=device, dtype=torch.bfloat16)
    weights = torch.randn(8, 1, device=device, dtype=torch.float32)
    fused_ops.assert_row_weighting_supported(rows, weights)

    with pytest.raises(RuntimeError, match="contiguous rows"):
        fused_ops.assert_row_weighting_supported(rows.t(), torch.randn(16, 1, device=device, dtype=torch.float32))

    with pytest.raises(RuntimeError, match=r"shape \[N, 1\]"):
        fused_ops.assert_row_weighting_supported(rows, torch.randn(8, device=device, dtype=torch.float32))

    with pytest.raises(RuntimeError, match="bfloat16 and float16"):
        fused_ops.assert_row_weighting_supported(rows.float(), weights)

    with pytest.raises(RuntimeError, match="float32 weights"):
        fused_ops.assert_row_weighting_supported(rows, weights.to(torch.bfloat16))
