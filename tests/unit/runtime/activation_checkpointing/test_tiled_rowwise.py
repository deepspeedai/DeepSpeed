# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import pytest
import torch

from deepspeed.runtime.activation_checkpointing.tiled_rowwise import plan_row_chunks
from deepspeed.runtime.activation_checkpointing.tiled_rowwise import tiled_rowwise_compute


def _fused_add_rms_norm(x, residual, weight, eps=1e-6):
    # Row-wise in the token axis: a residual add and a (1 + weight) RMSNorm, both in fp32, two outputs.
    dtype = torch.float32 if x.dtype in (torch.bfloat16, torch.float16) else x.dtype  # fp64 stays fp64 for gradcheck
    summed = x.to(dtype) + residual.to(dtype)
    rstd = torch.rsqrt(summed.pow(2).mean(-1, keepdim=True) + eps)
    out = summed * rstd * (1.0 + weight.to(dtype))
    return out.to(x.dtype), summed.to(x.dtype)


def _inputs(dtype, rows=37, hidden=16, seed=0):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(1, rows, hidden, generator=g).to(dtype).requires_grad_()
    r = torch.randn(1, rows, hidden, generator=g).to(dtype).requires_grad_()
    w = torch.nn.Parameter(torch.randn(hidden, generator=g).to(dtype) * 0.1)
    return x, r, w


def _run(fn, x, r, w, budget=None, **kw):
    x, r, w = x.detach().clone().requires_grad_(), r.detach().clone().requires_grad_(), torch.nn.Parameter(
        w.detach().clone())
    if budget is None:
        out, summed = fn(x, r, w)
    else:
        out, summed = tiled_rowwise_compute(fn, x, r, shared=(w, ), token_budget=budget, **kw)
    g = torch.Generator().manual_seed(7)
    (out.float() * torch.randn(out.shape, generator=g)).sum().backward(retain_graph=True)
    (summed.float() * torch.randn(summed.shape, generator=g)).sum().backward()
    return out, summed, x.grad, r.grad, w.grad


def test_plan_row_chunks():
    assert plan_row_chunks(10, 4) == [(0, 4), (4, 8), (8, 10)]
    assert plan_row_chunks(8, 4) == [(0, 4), (4, 8)]
    assert plan_row_chunks(3, 4) == [(0, 3)]
    with pytest.raises(ValueError, match="positive"):
        plan_row_chunks(3, 0)


@pytest.mark.parametrize("budget", [1, 5, 16, 36])
def test_forward_is_bit_identical_and_grads_match(budget):
    x, r, w = _inputs(torch.bfloat16)
    ref = _run(_fused_add_rms_norm, x, r, w)
    got = _run(_fused_add_rms_norm, x, r, w, budget=budget)
    assert torch.equal(got[0], ref[0]) and torch.equal(got[1], ref[1])
    assert torch.equal(got[2], ref[2]) and torch.equal(got[3], ref[3])  # input grads are per row: exact
    # The weight grad is a sum over chunks. Accumulated in fp32 it is one bf16 rounding from the reference; summed in
    # bf16 every chunk adds a rounding, so that path only gets a loose bound.
    fp32 = _run(_fused_add_rms_norm, x, r, w, budget=budget, shared_grad_dtype=torch.float32)
    torch.testing.assert_close(fp32[4].float(), ref[4].float(), rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(got[4].float(), ref[4].float(), rtol=0.3, atol=0.1)


def test_fp32_shared_grad_accumulation_is_closer_than_bf16():
    x, r, w = _inputs(torch.bfloat16, rows=257, hidden=32)
    ref = _run(_fused_add_rms_norm, x.double(), r.double(), w.double())[4]
    bf16 = _run(_fused_add_rms_norm, x, r, w, budget=7)[4]
    fp32 = _run(_fused_add_rms_norm, x, r, w, budget=7, shared_grad_dtype=torch.float32)[4]
    assert fp32.dtype == bf16.dtype == torch.bfloat16
    assert (fp32.double() - ref).abs().max() <= (bf16.double() - ref).abs().max()


def test_gradcheck_float64():
    x, r, w = _inputs(torch.float64, rows=9, hidden=4)

    def fn(a, b, weight):
        return tiled_rowwise_compute(_fused_add_rms_norm, a, b, shared=(weight, ), token_budget=4)

    assert torch.autograd.gradcheck(fn, (x, r, w), eps=1e-6, atol=1e-5)


def test_single_chunk_calls_fn_directly():
    calls = []

    def fn(a):
        calls.append(a.shape[0])
        return a * 2

    x = torch.ones(5, 3, requires_grad=True)
    out = tiled_rowwise_compute(fn, x, token_budget=8, token_dim=0)
    assert calls == [5] and out.grad_fn is not None and type(out.grad_fn).__name__ != "_TiledRowwiseBackward"


def test_output_without_grad_and_input_without_grad():
    x, r, w = _inputs(torch.float32)
    r = r.detach()
    out, summed = tiled_rowwise_compute(_fused_add_rms_norm, x, r, shared=(w, ), token_budget=5)
    out.sum().backward()  # summed receives no gradient
    assert x.grad is not None and w.grad is not None and r.grad is None


def test_rejects_mismatched_rows_and_bad_returns():
    with pytest.raises(ValueError, match="same extent"):
        tiled_rowwise_compute(lambda a, b: a + b[:, :2], torch.ones(1, 4, 2), torch.ones(1, 3, 2), token_budget=2)
    with pytest.raises(TypeError, match="tensor or a tuple"):
        tiled_rowwise_compute(lambda a: [a], torch.ones(1, 4, 2), token_budget=2)


@pytest.mark.parametrize("budget", [1, 5, 36])
def test_unchunked_forward_runs_once_and_matches_the_chunked_forward(budget):
    x, r, w = _inputs(torch.bfloat16)
    calls = []

    def fn(a, b, weight):
        calls.append((torch.is_grad_enabled(), a.shape[-2]))
        return _fused_add_rms_norm(a, b, weight)

    ref = _run(_fused_add_rms_norm, x, r, w)
    got = _run(fn, x, r, w, budget=budget, shared_grad_dtype=torch.float32, chunk_forward=False)
    assert calls[0] == (False, x.shape[-2])  # one whole-row forward under no_grad
    assert all(grad_enabled for grad_enabled, _ in calls[1:])  # then the recompute, chunk by chunk, with grad
    assert max(rows for _, rows in calls[1:]) <= budget
    assert torch.equal(got[0], ref[0]) and torch.equal(got[1], ref[1])
    assert torch.equal(got[2], ref[2]) and torch.equal(got[3], ref[3])
    torch.testing.assert_close(got[4].float(), ref[4].float(), rtol=1e-2, atol=1e-2)


def test_gradcheck_float64_with_an_unchunked_forward():
    x, r, w = _inputs(torch.float64, rows=9, hidden=4)

    def fn(a, b, weight):
        return tiled_rowwise_compute(_fused_add_rms_norm, a, b, shared=(weight, ), token_budget=4, chunk_forward=False)

    assert torch.autograd.gradcheck(fn, (x, r, w), eps=1e-6, atol=1e-5)
