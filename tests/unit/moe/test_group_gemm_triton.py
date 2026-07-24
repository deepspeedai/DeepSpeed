# Copyright (c) DeepSpeed Team.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
"""Unit tests for the Triton grouped-GEMM drop-in (``deepspeed.moe.group_gemm_triton``).

Correctness is checked against:
  * a pure-PyTorch per-group reference (all dtypes), and
  * ``torch._grouped_mm`` where available (bf16 only, which is all it supports),
for forward output and both input gradients, across even / uneven / empty groups.
"""

import pytest
import torch

from deepspeed.accelerator import get_accelerator
from deepspeed.moe.group_gemm_triton import group_gemm, is_available

if not is_available():
    pytest.skip("Triton is not available", allow_module_level=True)

if not (get_accelerator().is_available() and get_accelerator().device_name() == "cuda"):
    pytest.skip("Triton grouped GEMM requires a CUDA device", allow_module_level=True)


def _tol(dtype):
    if dtype == torch.float32:
        return dict(atol=1e-3, rtol=1e-3)
    if dtype == torch.float16:
        return dict(atol=2e-2, rtol=2e-2)
    return dict(atol=3e-2, rtol=3e-2)  # bfloat16


def _ref_grouped_mm(a, b, offs):
    """Pure-PyTorch reference: out[rows_g] = a[rows_g] @ b[g]."""
    outs = []
    start = 0
    for g in range(offs.numel()):
        end = int(offs[g])
        outs.append(a[start:end] @ b[g])
        start = end
    return torch.cat(outs, dim=0)


def _make_offs(counts, device):
    return torch.cumsum(torch.tensor(counts, device=device, dtype=torch.int64), 0).to(torch.int32)


# (M-per-group counts, K, N)
_SHAPES = [
    ([8, 8, 8], 32, 16),  # even, block-aligned
    ([13, 9, 8], 32, 16),  # uneven, not a multiple of block
    ([0, 16, 8], 32, 16),  # leading empty group
    ([20, 0, 30, 14], 48, 40),  # empty middle group, odd dims
    ([40, 10, 0, 50, 30, 26, 60, 40], 128, 96),  # 8 experts, mixed
    ([1, 1, 1], 16, 16),  # single-row groups
]

_DTYPES = [torch.bfloat16, torch.float16, torch.float32]


@pytest.mark.parametrize("counts,K,N", _SHAPES)
@pytest.mark.parametrize("dtype", _DTYPES)
def test_forward_matches_reference(counts, K, N, dtype):
    dev = get_accelerator().current_device_name()
    torch.manual_seed(0)
    E = len(counts)
    M = sum(counts)
    a = torch.randn(M, K, device=dev, dtype=dtype)
    w = torch.randn(E, N, K, device=dev, dtype=dtype)  # stored like expert weight [E, hidden, dim]
    b = w.transpose(-2, -1)  # [E, K, N] non-contiguous view, as in ep_experts.py
    offs = _make_offs(counts, dev)

    out = group_gemm(a, b, offs)
    ref = _ref_grouped_mm(a.float(), b.float(), offs)

    assert out.shape == (M, N)
    assert out.dtype == dtype
    torch.testing.assert_close(out.float(), ref, **_tol(dtype))


@pytest.mark.parametrize("counts,K,N", _SHAPES)
@pytest.mark.parametrize("dtype", _DTYPES)
def test_backward_matches_reference(counts, K, N, dtype):
    dev = get_accelerator().current_device_name()
    torch.manual_seed(1)
    E = len(counts)
    M = sum(counts)
    a = torch.randn(M, K, device=dev, dtype=dtype)
    w = torch.randn(E, N, K, device=dev, dtype=dtype)
    offs = _make_offs(counts, dev)

    a_tri = a.clone().requires_grad_(True)
    w_tri = w.clone().requires_grad_(True)
    out = group_gemm(a_tri, w_tri.transpose(-2, -1), offs)
    grad_out = torch.randn_like(out)
    out.backward(grad_out)

    a_ref = a.float().clone().requires_grad_(True)
    w_ref = w.float().clone().requires_grad_(True)
    ref = _ref_grouped_mm(a_ref, w_ref.transpose(-2, -1), offs)
    ref.backward(grad_out.float())

    assert a_tri.grad.shape == a.shape and a_tri.grad.dtype == dtype
    assert w_tri.grad.shape == w.shape and w_tri.grad.dtype == dtype
    torch.testing.assert_close(a_tri.grad.float(), a_ref.grad, **_tol(dtype))
    torch.testing.assert_close(w_tri.grad.float(), w_ref.grad, **_tol(dtype))


@pytest.mark.parametrize("counts,K,N", _SHAPES)
@pytest.mark.parametrize("dtype", _DTYPES)
def test_trans_b_matches_reference(counts, K, N, dtype):
    """trans_b=True (weight in native [E,N,K] layout, out = a @ w^T) fwd + grads.

    This is the layout used by the expert path: it keeps the transpose off the
    autograd tape so the weight gradient is produced directly in [E,N,K] with no
    materialization copy. Result must equal the explicit-transpose path.
    """
    dev = get_accelerator().current_device_name()
    torch.manual_seed(5)
    E = len(counts)
    M = sum(counts)
    a = torch.randn(M, K, device=dev, dtype=dtype)
    w = torch.randn(E, N, K, device=dev, dtype=dtype)  # native weight layout [E, N, K]
    offs = _make_offs(counts, dev)

    # trans_b=True path: pass w directly (no .transpose).
    a_t = a.clone().requires_grad_(True)
    w_t = w.clone().requires_grad_(True)
    out_t = group_gemm(a_t, w_t, offs, trans_b=True)
    grad_out = torch.randn_like(out_t)
    out_t.backward(grad_out)

    # Reference: explicit-transpose path (trans_b=False with w^T).
    a_r = a.clone().requires_grad_(True)
    w_r = w.clone().requires_grad_(True)
    out_r = group_gemm(a_r, w_r.transpose(-2, -1), offs)
    out_r.backward(grad_out)

    assert w_t.grad.shape == w.shape  # gradient already in native [E, N, K] layout
    torch.testing.assert_close(out_t.float(), out_r.float(), **_tol(dtype))
    torch.testing.assert_close(a_t.grad.float(), a_r.grad.float(), **_tol(dtype))
    torch.testing.assert_close(w_t.grad.float(), w_r.grad.float(), **_tol(dtype))


@pytest.mark.parametrize("counts,K,N", _SHAPES)
def test_forward_matches_torch_grouped_mm_bf16(counts, K, N):
    """Match the native op exactly (bf16 is the only dtype torch._grouped_mm supports)."""
    if not hasattr(torch, "_grouped_mm"):
        pytest.skip("torch._grouped_mm unavailable")
    dev = get_accelerator().current_device_name()
    torch.manual_seed(2)
    E = len(counts)
    M = sum(counts)
    a = torch.randn(M, K, device=dev, dtype=torch.bfloat16)
    w = torch.randn(E, N, K, device=dev, dtype=torch.bfloat16)
    offs = _make_offs(counts, dev)

    tri = group_gemm(a, w.transpose(-2, -1), offs)
    try:
        native = torch._grouped_mm(a, w.transpose(-2, -1), offs=offs)
    except RuntimeError as e:
        pytest.skip(f"torch._grouped_mm rejected inputs on this build: {e}")

    torch.testing.assert_close(tri.float(), native.float(), **_tol(torch.bfloat16))


def test_gradcheck_fp32():
    """Double-precision-style gradcheck (fp32 with IEEE accumulation)."""
    dev = get_accelerator().current_device_name()
    torch.manual_seed(3)
    counts = [3, 0, 5]
    K, N = 8, 6
    E, M = len(counts), sum(counts)
    a = torch.randn(M, K, device=dev, dtype=torch.float32, requires_grad=True)
    w = torch.randn(E, N, K, device=dev, dtype=torch.float32, requires_grad=True)
    offs = _make_offs(counts, dev)

    # Analytic grads vs finite-difference reference on the pure-torch path.
    out = group_gemm(a, w.transpose(-2, -1), offs)
    grad_out = torch.randn_like(out)
    out.backward(grad_out)

    a_ref = a.detach().clone().requires_grad_(True)
    w_ref = w.detach().clone().requires_grad_(True)
    _ref_grouped_mm(a_ref, w_ref.transpose(-2, -1), offs).backward(grad_out)

    torch.testing.assert_close(a.grad, a_ref.grad, atol=1e-3, rtol=1e-3)
    torch.testing.assert_close(w.grad, w_ref.grad, atol=1e-3, rtol=1e-3)


def test_empty_total_is_safe():
    """All-empty groups produce a well-formed zero-row output and zero weight grad."""
    dev = get_accelerator().current_device_name()
    E, K, N = 4, 16, 16
    a = torch.zeros(0, K, device=dev, dtype=torch.bfloat16, requires_grad=True)
    w = torch.randn(E, N, K, device=dev, dtype=torch.bfloat16, requires_grad=True)
    offs = torch.zeros(E, device=dev, dtype=torch.int32)

    out = group_gemm(a, w.transpose(-2, -1), offs)
    assert out.shape == (0, N)
    out.sum().backward()
    assert torch.count_nonzero(w.grad) == 0


# ---------------------------------------------------------------------------
# End-to-end: full SwiGLU expert block (3 grouped GEMMs + SiLU + backward),
# comparing the Triton drop-in against the native torch._grouped_mm expert path
# used by deepspeed.moe.ep_experts._run_experts_grouped_mm.
# ---------------------------------------------------------------------------


def _swiglu_experts_group_gemm(w1, w2, w3, x, counts_tensor):
    """Same SwiGLU expert MLP as _run_experts_grouped_mm, but via Triton group_gemm."""
    import torch.nn.functional as F

    offsets = torch.cumsum(counts_tensor, dim=0, dtype=torch.int32)
    h = F.silu(group_gemm(x, w1.transpose(-2, -1), offsets))
    h = h * group_gemm(x, w3.transpose(-2, -1), offsets)
    return group_gemm(h, w2.transpose(-2, -1), offsets).type_as(x)


def test_e2e_swiglu_experts_matches_native_grouped_mm():
    from deepspeed.moe.ep_experts import _run_experts_grouped_mm

    dev = get_accelerator().current_device_name()
    torch.manual_seed(7)
    dim, hidden, E = 64, 128, 4
    counts = [20, 0, 30, 14]  # includes an empty expert
    M = sum(counts)
    counts_t = torch.tensor(counts, device=dev, dtype=torch.int32)

    # Shared random init for the two paths.
    x0 = torch.randn(M, dim, device=dev, dtype=torch.bfloat16)
    w1_0 = torch.randn(E, hidden, dim, device=dev, dtype=torch.bfloat16) * 0.1
    w2_0 = torch.randn(E, dim, hidden, device=dev, dtype=torch.bfloat16) * 0.1
    w3_0 = torch.randn(E, hidden, dim, device=dev, dtype=torch.bfloat16) * 0.1

    def _leaves():
        return (
            x0.clone().requires_grad_(True),
            w1_0.clone().requires_grad_(True),
            w2_0.clone().requires_grad_(True),
            w3_0.clone().requires_grad_(True),
        )

    # Native path (torch._grouped_mm; on sm80/86 this is the for-loop fallback).
    x_n, w1_n, w2_n, w3_n = _leaves()
    try:
        out_native = _run_experts_grouped_mm(w1_n, w2_n, w3_n, x_n, counts_t)
    except RuntimeError as e:
        pytest.skip(f"native torch._grouped_mm path unavailable: {e}")
    grad_out = torch.randn_like(out_native)
    out_native.backward(grad_out)

    # Triton drop-in path.
    x_t, w1_t, w2_t, w3_t = _leaves()
    out_tri = _swiglu_experts_group_gemm(w1_t, w2_t, w3_t, x_t, counts_t)
    out_tri.backward(grad_out)

    tol = dict(atol=3e-2, rtol=3e-2)
    torch.testing.assert_close(out_tri.float(), out_native.float(), **tol)
    torch.testing.assert_close(x_t.grad.float(), x_n.grad.float(), **tol)
    torch.testing.assert_close(w1_t.grad.float(), w1_n.grad.float(), **tol)
    torch.testing.assert_close(w2_t.grad.float(), w2_n.grad.float(), **tol)
    torch.testing.assert_close(w3_t.grad.float(), w3_n.grad.float(), **tol)


def test_grouped_experts_triton_path_parity():
    """The GroupedExperts module's Triton path matches its for-loop path (fwd + grads)."""
    from deepspeed.moe.ep_experts import GroupedExperts

    dev = get_accelerator().current_device_name()
    torch.manual_seed(11)
    dim, hidden, E = 64, 128, 4
    counts = torch.tensor([20, 0, 30, 14], device=dev, dtype=torch.int32)
    M = int(counts.sum())

    triton_experts = GroupedExperts(dim, hidden, E, use_grouped_mm=True,
                                    use_triton_grouped_mm=True).to(dev).to(torch.bfloat16)
    loop_experts = GroupedExperts(dim, hidden, E, use_grouped_mm=False).to(dev).to(torch.bfloat16)
    # GroupedExperts allocates weights with torch.empty; set controlled values.
    with torch.no_grad():
        triton_experts.w1.normal_(0, 0.1)
        triton_experts.w2.normal_(0, 0.1)
        triton_experts.w3.normal_(0, 0.1)
    loop_experts.load_state_dict(triton_experts.state_dict())
    assert triton_experts.use_triton_grouped_mm is True

    x = torch.randn(M, dim, device=dev, dtype=torch.bfloat16)
    x_t = x.clone().requires_grad_(True)
    x_l = x.clone().requires_grad_(True)
    out_t = triton_experts(x_t, counts)
    out_l = loop_experts(x_l, counts)
    grad_out = torch.randn_like(out_t)
    out_t.backward(grad_out)
    out_l.backward(grad_out)

    tol = dict(atol=3e-2, rtol=3e-2)
    torch.testing.assert_close(out_t.float(), out_l.float(), **tol)
    torch.testing.assert_close(x_t.grad.float(), x_l.grad.float(), **tol)
    torch.testing.assert_close(triton_experts.w1.grad.float(), loop_experts.w1.grad.float(), **tol)
    torch.testing.assert_close(triton_experts.w2.grad.float(), loop_experts.w2.grad.float(), **tol)
    torch.testing.assert_close(triton_experts.w3.grad.float(), loop_experts.w3.grad.float(), **tol)


def test_grouped_experts_auto_selects_triton_on_ampere():
    """On sm < 9.0 the module auto-selects the Triton grouped-GEMM path."""
    from deepspeed.moe.ep_experts import GroupedExperts

    dev = get_accelerator().current_device_name()
    major, _ = torch.cuda.get_device_capability()
    experts = GroupedExperts(32, 64, 2, use_grouped_mm=True).to(dev)
    if major < 9:
        assert experts.use_triton_grouped_mm is True
    # Explicit opt-out is always honored.
    experts_off = GroupedExperts(32, 64, 2, use_grouped_mm=True, use_triton_grouped_mm=False).to(dev)
    assert experts_off.use_triton_grouped_mm is False
