# Copyright (c) DeepSpeed Team.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
"""Unit tests for the Triton grouped-GEMM drop-in (``deepspeed.ops.triton_ops.group_gemm_triton``).

Correctness is checked against:
  * a pure-PyTorch per-group reference (all dtypes), and
  * ``torch._grouped_mm`` where available (bf16 only, which is all it supports),
for forward output and both input gradients, across even / uneven / empty groups.
"""

import pytest
import torch

from deepspeed.accelerator import get_accelerator
from deepspeed.ops.triton_ops import is_triton_available
from deepspeed.ops.triton_ops.group_gemm_triton import group_gemm_triton
from deepspeed.ops.triton_ops.group_gemm_triton import _group_meta, _GROUP_META_BLOCK

if not is_triton_available():
    pytest.skip("Triton is not available", allow_module_level=True)

if not (get_accelerator().is_available() and get_accelerator().device_name() == "cuda"):
    pytest.skip("Triton grouped GEMM requires a CUDA device", allow_module_level=True)


def _tol(dtype):
    if dtype == torch.float32:
        # fp32 uses Triton's default tl.dot precision (TF32 on Ampere), compared
        # against the full-precision torch reference, so use a TF32-level tol.
        return dict(atol=2e-2, rtol=2e-2)
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


def _ref_group_meta(offs):
    """Reference for _group_meta: m_start = exclusive-prefix, m_size = per-group count."""
    offs_i = offs.to(torch.int32)
    zero = torch.zeros(1, dtype=torch.int32, device=offs_i.device)
    m_start = torch.cat([zero, offs_i[:-1]])
    m_size = offs_i - m_start
    return m_start, m_size


# ---------------------------------------------------------------------------
# _group_meta Triton kernel (per-group m-start / m-size from cumulative offs)
# ---------------------------------------------------------------------------

_GROUP_META_CASES = [
    [16, 24, 40],  # simple
    [0, 16, 24],  # leading empty group
    [16, 40, 40],  # trailing empty group
    [0, 0, 24],  # consecutive empty groups
    [32],  # single group (E == 1)
    [8, 8, 8, 8],  # even
    [5, 0, 0, 7, 0],  # multiple empties
]


@pytest.mark.parametrize("cumulative", _GROUP_META_CASES)
def test_group_meta_matches_reference(cumulative):
    dev = get_accelerator().current_device_name()
    offs = torch.tensor(cumulative, device=dev, dtype=torch.int32)
    m_start, m_size = _group_meta(offs)
    ref_start, ref_size = _ref_group_meta(offs)

    assert m_start.dtype == torch.int32 and m_size.dtype == torch.int32
    assert m_start.shape == offs.shape and m_size.shape == offs.shape
    torch.testing.assert_close(m_start, ref_start, atol=0, rtol=0)
    torch.testing.assert_close(m_size, ref_size, atol=0, rtol=0)
    # m_start[0] must be 0 and m_size must sum to the total (offs[-1]).
    assert int(m_start[0]) == 0
    assert int(m_size.sum()) == int(offs[-1])


def test_group_meta_large_E_spans_multiple_blocks():
    """E larger than the fixed BLOCK exercises the multi-block grid path."""
    dev = get_accelerator().current_device_name()
    E = _GROUP_META_BLOCK * 3 + 7  # forces grid > 1, with a partial last block
    torch.manual_seed(0)
    counts = torch.randint(0, 5, (E, ), device=dev, dtype=torch.int32)
    offs = torch.cumsum(counts, 0).to(torch.int32)
    m_start, m_size = _group_meta(offs)
    ref_start, ref_size = _ref_group_meta(offs)
    torch.testing.assert_close(m_start, ref_start, atol=0, rtol=0)
    torch.testing.assert_close(m_size, ref_size, atol=0, rtol=0)


# (M-per-group counts, K, N)
_SHAPES = [
    ([8, 8, 8], 32, 16),  # even, block-aligned
    ([13, 9, 8], 32, 16),  # uneven, not a multiple of block
    ([0, 16, 8], 32, 16),  # leading empty group
    ([20, 0, 30, 14], 48, 40),  # empty middle group, odd dims
    ([40, 10, 0, 50, 30, 26, 60, 40], 128, 96),  # 8 experts, mixed
    ([1, 1, 1], 16, 16),  # single-row groups
]

# bfloat16 requires compute capability >= 8.0 (Ampere+); include it only there.
_DTYPES = [torch.float16, torch.float32]
if torch.cuda.get_device_capability()[0] >= 8:  #ignore-cuda
    _DTYPES = [torch.bfloat16] + _DTYPES


@pytest.mark.parametrize("counts,K,N", _SHAPES)
@pytest.mark.parametrize("dtype", _DTYPES)
def test_forward_matches_reference(counts, K, N, dtype):
    dev = get_accelerator().current_device_name()
    torch.manual_seed(0)
    E = len(counts)
    M = sum(counts)
    a = torch.randn(M, K, device=dev, dtype=dtype)
    b = torch.randn(E, K, N, device=dev, dtype=dtype)  # contiguous [E, K, N], trans_b=False
    offs = _make_offs(counts, dev)

    out = group_gemm_triton(a, b, offs)
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
    b = torch.randn(E, K, N, device=dev, dtype=dtype)  # contiguous [E, K, N], trans_b=False
    offs = _make_offs(counts, dev)

    a_tri = a.clone().requires_grad_(True)
    b_tri = b.clone().requires_grad_(True)
    out = group_gemm_triton(a_tri, b_tri, offs)
    grad_out = torch.randn_like(out)
    out.backward(grad_out)

    a_ref = a.float().clone().requires_grad_(True)
    b_ref = b.float().clone().requires_grad_(True)
    ref = _ref_grouped_mm(a_ref, b_ref, offs)
    ref.backward(grad_out.float())

    assert a_tri.grad.shape == a.shape and a_tri.grad.dtype == dtype
    assert b_tri.grad.shape == b.shape and b_tri.grad.dtype == dtype
    torch.testing.assert_close(a_tri.grad.float(), a_ref.grad, **_tol(dtype))
    torch.testing.assert_close(b_tri.grad.float(), b_ref.grad, **_tol(dtype))


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
    out_t = group_gemm_triton(a_t, w_t, offs, trans_b=True)
    grad_out = torch.randn_like(out_t)
    out_t.backward(grad_out)

    # Reference: pure-torch a @ w^T per group (w in native [E, N, K] layout).
    a_r = a.clone().requires_grad_(True)
    w_r = w.clone().requires_grad_(True)
    out_r = _ref_grouped_mm(a_r, w_r.transpose(-2, -1), offs)
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

    tri = group_gemm_triton(a, w, offs, trans_b=True)
    try:
        native = torch._grouped_mm(a, w.transpose(-2, -1), offs=offs)
    except RuntimeError as e:
        pytest.skip(f"torch._grouped_mm rejected inputs on this build: {e}")

    torch.testing.assert_close(tri.float(), native.float(), **_tol(torch.bfloat16))


def test_gradcheck_fp32():
    """fp32 grad parity vs the pure-torch reference (Triton default tl.dot precision)."""
    dev = get_accelerator().current_device_name()
    torch.manual_seed(3)
    counts = [3, 0, 5]
    K, N = 8, 6
    E, M = len(counts), sum(counts)
    a = torch.randn(M, K, device=dev, dtype=torch.float32, requires_grad=True)
    w = torch.randn(E, N, K, device=dev, dtype=torch.float32, requires_grad=True)
    offs = _make_offs(counts, dev)

    # Analytic grads vs finite-difference reference on the pure-torch path.
    out = group_gemm_triton(a, w, offs, trans_b=True)
    grad_out = torch.randn_like(out)
    out.backward(grad_out)

    a_ref = a.detach().clone().requires_grad_(True)
    w_ref = w.detach().clone().requires_grad_(True)
    _ref_grouped_mm(a_ref, w_ref.transpose(-2, -1), offs).backward(grad_out)

    torch.testing.assert_close(a.grad, a_ref.grad, **_tol(torch.float32))
    torch.testing.assert_close(w.grad, w_ref.grad, **_tol(torch.float32))


def test_noncontiguous_mat_b_is_rejected():
    """A non-contiguous (transposed-view) mat_b must be rejected in favor of trans_b."""
    dev = get_accelerator().current_device_name()
    a = torch.randn(16, 8, device=dev, dtype=torch.bfloat16)
    w = torch.randn(2, 6, 8, device=dev, dtype=torch.bfloat16)  # [E, N, K]
    offs = _make_offs([8, 8], dev)
    b_view = w.transpose(-2, -1)  # non-contiguous [E, K, N] view
    assert not b_view.is_contiguous()
    with pytest.raises(ValueError, match="contiguous"):
        group_gemm_triton(a, b_view, offs)


def test_empty_total_is_safe():
    """All-empty groups produce a well-formed zero-row output and zero weight grad."""
    dev = get_accelerator().current_device_name()
    E, K, N = 4, 16, 16
    a = torch.zeros(0, K, device=dev, dtype=torch.bfloat16, requires_grad=True)
    w = torch.randn(E, N, K, device=dev, dtype=torch.bfloat16, requires_grad=True)
    offs = torch.zeros(E, device=dev, dtype=torch.int32)

    out = group_gemm_triton(a, w, offs, trans_b=True)
    assert out.shape == (0, N)
    out.sum().backward()
    assert torch.count_nonzero(w.grad) == 0


# ---------------------------------------------------------------------------
# End-to-end: full SwiGLU expert block (3 grouped GEMMs + SiLU + backward),
# comparing the Triton drop-in against the native torch._grouped_mm expert path
# used by deepspeed.moe.ep_experts._run_experts_grouped_mm.
# ---------------------------------------------------------------------------


def _swiglu_experts_group_gemm(w1, w2, w3, x, counts_tensor):
    """Same SwiGLU expert MLP as _run_experts_grouped_mm, but via Triton group_gemm_triton."""
    import torch.nn.functional as F

    offsets = torch.cumsum(counts_tensor, dim=0, dtype=torch.int32)
    h = F.silu(group_gemm_triton(x, w1, offsets, trans_b=True))
    h = h * group_gemm_triton(x, w3, offsets, trans_b=True)
    return group_gemm_triton(h, w2, offsets, trans_b=True).type_as(x)


def test_e2e_swiglu_experts_matches_native_grouped_mm():
    from deepspeed.moe.ep_experts import _run_experts_grouped_mm

    dev = get_accelerator().current_device_name()
    torch.manual_seed(7)
    dim, hidden, E = 64, 128, 4
    counts = [20, 0, 30, 14]  # includes an empty expert
    M = sum(counts)
    counts_t = torch.tensor(counts, device=dev, dtype=torch.int32)

    # Shared random init for the two paths.
    # Bounded [-1, 1] inputs keep the SwiGLU activations small so the two paths' differing
    # bf16 accumulation orders stay within tolerance (unbounded randn tails blow up weight grads).
    x0 = torch.empty(M, dim, device=dev, dtype=torch.bfloat16).uniform_(-1, 1)
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

    # use_grouped_mm=True auto-selects the Triton path on sm < 9.0 (e.g. A6000).
    triton_experts = GroupedExperts(dim, hidden, E, use_grouped_mm=True).to(dev).to(torch.bfloat16)
    if not triton_experts.use_triton_grouped_mm:
        pytest.skip("Triton grouped-GEMM path not selected on this device (sm >= 9.0)")
    loop_experts = GroupedExperts(dim, hidden, E, use_grouped_mm=False).to(dev).to(torch.bfloat16)
    # GroupedExperts allocates weights with torch.empty; set controlled values.
    with torch.no_grad():
        triton_experts.w1.normal_(0, 0.1)
        triton_experts.w2.normal_(0, 0.1)
        triton_experts.w3.normal_(0, 0.1)
    loop_experts.load_state_dict(triton_experts.state_dict())
    assert triton_experts.use_triton_grouped_mm is True

    # Bounded [-1, 1] inputs keep activations small so the two paths agree within tolerance.
    x = torch.empty(M, dim, device=dev, dtype=torch.bfloat16).uniform_(-1, 1)
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
    """On sm < 9.0 the module auto-selects Triton; disable_triton_grouped_mm opts out."""
    from deepspeed.moe.ep_experts import GroupedExperts

    dev = get_accelerator().current_device_name()
    major, _ = torch.cuda.get_device_capability()  #ignore-cuda

    experts = GroupedExperts(32, 64, 2, use_grouped_mm=True).to(dev)
    if major < 9:
        assert experts.use_triton_grouped_mm is True
    else:
        assert experts.use_triton_grouped_mm is False  # sm90+ uses native torch._grouped_mm

    # Config override disables the Triton path regardless of device.
    experts_off = GroupedExperts(32, 64, 2, use_grouped_mm=True, disable_triton_grouped_mm=True).to(dev)
    assert experts_off.use_triton_grouped_mm is False


def test_expert_offset_exceeds_int32():
    """Expert base offsets past 2**31 elements must be computed in int64.

    ``b_base = b_ptr + selected * stride_be`` walks ``E * K * N`` elements. With
    a large enough expert weight that product overflows int32 and wraps to a
    negative offset, so the kernel reads out of bounds and faults. Sizes here put
    the last expert at ~2.2e9 elements, just past the int32 limit.
    """
    K = N = 8192
    num_experts = 34
    rows_per_expert = 8
    dtype = torch.bfloat16

    stride_be = K * N
    assert (num_experts - 1) * stride_be > 2**31 - 1, "sizes no longer exercise the overflow"

    needed = num_experts * stride_be * torch.finfo(dtype).bits // 8
    if get_accelerator().available_memory() < 2 * needed:
        pytest.skip(f"needs ~{2 * needed / 2**30:.1f} GiB of free device memory")

    dev = get_accelerator().current_device_name()
    m = num_experts * rows_per_expert
    a = torch.randn(m, K, dtype=dtype, device=dev)
    b = torch.randn(num_experts, K, N, dtype=dtype, device=dev)
    offs = _make_offs([rows_per_expert] * num_experts, dev)

    out = group_gemm_triton(a, b, offs)

    # Check the last expert specifically -- it carries the largest offset.
    last = a[-rows_per_expert:] @ b[-1]
    torch.testing.assert_close(out[-rows_per_expert:].float(), last.float(), **_tol(dtype))
