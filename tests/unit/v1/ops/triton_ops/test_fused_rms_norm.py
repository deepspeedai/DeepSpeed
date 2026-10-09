# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team
"""Compare fused RMSNorm with the HF eager expression."""

import copy
import functools
import importlib
import types

import pytest
import torch

from deepspeed.accelerator import get_accelerator
from deepspeed.ops.triton_ops import fused_rms_norm

QWEN3_MOE_RMS_NORM = "transformers.models.qwen3_moe.modeling_qwen3_moe.Qwen3MoeRMSNorm"


def _fused_engine_available():
    accelerator = get_accelerator()
    return (accelerator.is_available() and accelerator.device_name().startswith("cuda")
            and fused_rms_norm.is_available())


def _device():
    return get_accelerator().current_device_name()


def _hf_rms_norm(hidden, weight, eps):
    input_dtype = hidden.dtype
    h = hidden.float()
    variance = h.pow(2).mean(-1, keepdim=True)
    h = h * torch.rsqrt(variance + eps)
    return weight * h.to(input_dtype)


def _gamma_before_cast_rms_norm(hidden, weight, eps):
    # The order GPT-OSS and recent Olmo2 releases use: the weight multiplies the FP32 value before the cast.
    input_dtype = hidden.dtype
    h = hidden.float()
    variance = h.pow(2).mean(-1, keepdim=True)
    h = h * torch.rsqrt(variance + eps)
    return (weight * h).to(input_dtype)


def _hf_class(qualified_name):
    module_name, _, class_name = qualified_name.rpartition(".")
    try:
        return getattr(importlib.import_module(module_name), class_name)
    except (ImportError, AttributeError):
        pytest.skip(f"{qualified_name} is not available in the installed transformers")


def _ordered_float_bits(tensor):
    bits = tensor.contiguous().view(torch.int16).to(torch.int32) & 0xffff
    sign = bits & 0x8000
    return torch.where(sign == 0, bits, 0x8000 - bits)


def _ulp_stats(actual, expected):
    assert actual.dtype == expected.dtype
    distances = (_ordered_float_bits(actual) - _ordered_float_bits(expected)).abs().flatten()
    if distances.numel() == 0:
        return {"max": 0, "median": 0.0, "frac_within_1": 1.0}
    return {
        "max": int(distances.max().item()),
        "median": float(distances.float().median().item()),
        "frac_within_1": float((distances <= 1).float().mean().item()),
    }


def _assert_ulp_close(actual, expected, *, max_ulp, min_frac_within_1, label):
    stats = _ulp_stats(actual, expected)
    message = (f"{label} ULP stats: max={stats['max']}, median={stats['median']}, "
               f"frac_within_1={stats['frac_within_1']}")
    assert stats["max"] <= max_ulp, message
    assert stats["frac_within_1"] >= min_frac_within_1, message


@pytest.mark.skipif(not _fused_engine_available(), reason="fused RMSNorm needs CUDA and Triton")
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize(
    "shape",
    [
        (0, 2048),
        (7, 128),
        (5, 2048),
        (3, 130),
        # Narrow norms run several rows per program; these cover partial last blocks and masked columns.
        (4099, 128),
        (2, 300, 32, 128),
        (1000, 96),
        (257, 64),
        (33, 16),
        (257, 7),
        (33, 15),
    ])
def test_fused_rms_norm_matches_hf_forward_and_backward(dtype, shape):
    device = _device()
    generator = torch.Generator(device=device).manual_seed(20260923)
    hidden = torch.randn(shape, device=device, dtype=dtype, generator=generator)
    weight = torch.randn((shape[-1], ), device=device, dtype=dtype, generator=generator)
    upstream = torch.randn(shape, device=device, dtype=dtype, generator=generator)

    eager_hidden = hidden.clone().requires_grad_(True)
    eager_weight = weight.clone().requires_grad_(True)
    eager_out = _hf_rms_norm(eager_hidden, eager_weight, 1e-6)
    eager_out.backward(upstream)

    fused_hidden = hidden.clone().requires_grad_(True)
    fused_weight = weight.clone().requires_grad_(True)
    fused_out = fused_rms_norm.fused_rms_norm(fused_hidden, fused_weight, 1e-6)
    fused_out.backward(upstream)

    _assert_ulp_close(fused_out, eager_out, max_ulp=2, min_frac_within_1=0.99, label="forward")
    _assert_ulp_close(fused_hidden.grad, eager_hidden.grad, max_ulp=8, min_frac_within_1=0.95, label="dx")
    _assert_ulp_close(fused_weight.grad, eager_weight.grad, max_ulp=8, min_frac_within_1=0.95, label="dgamma")


@pytest.mark.skipif(not _fused_engine_available(), reason="fused RMSNorm needs CUDA and Triton")
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_fused_rms_norm_width_one_has_nonzero_input_gradient(dtype):
    device = _device()
    eps = 0.25
    hidden = torch.linspace(0.25, 0.75, 257, device=device).to(dtype).reshape(-1, 1)
    weight = torch.ones((1, ), dtype=dtype, device=device)
    upstream = torch.ones_like(hidden)

    eager_hidden = hidden.clone().requires_grad_(True)
    eager_weight = weight.clone().requires_grad_(True)
    eager_out = _hf_rms_norm(eager_hidden, eager_weight, eps)
    eager_out.backward(upstream)

    fused_hidden = hidden.clone().requires_grad_(True)
    fused_weight = weight.clone().requires_grad_(True)
    fused_out = fused_rms_norm.fused_rms_norm(fused_hidden, fused_weight, eps)
    fused_out.backward(upstream)

    # Before gamma, d(x / sqrt(x**2 + eps))/dx = eps / (x**2 + eps)**1.5.
    reference_dx = (eps / (hidden.double().square() + eps).pow(1.5)).to(dtype)
    assert torch.all(reference_dx > 0)
    torch.testing.assert_close(fused_hidden.grad, reference_dx, rtol=2 * torch.finfo(dtype).eps, atol=0)
    _assert_ulp_close(fused_out, eager_out, max_ulp=2, min_frac_within_1=0.99, label="width-one forward")
    _assert_ulp_close(fused_hidden.grad, eager_hidden.grad, max_ulp=8, min_frac_within_1=0.95, label="width-one dx")
    _assert_ulp_close(fused_weight.grad,
                      eager_weight.grad,
                      max_ulp=8,
                      min_frac_within_1=0.95,
                      label="width-one dgamma")


@pytest.mark.skipif(not _fused_engine_available(), reason="fused RMSNorm needs CUDA and Triton")
@pytest.mark.parametrize("width", [128, 2048])
def test_fused_rms_norm_weight_gradient_is_deterministic(width):
    device = _device()
    generator = torch.Generator(device=device).manual_seed(7)
    hidden = torch.randn((8192, width), device=device, dtype=torch.bfloat16, generator=generator)
    weight = torch.randn((width, ), device=device, dtype=torch.bfloat16, generator=generator)
    upstream = torch.randn((8192, width), device=device, dtype=torch.bfloat16, generator=generator)
    grads = []
    for _ in range(2):
        leaf_hidden = hidden.clone().requires_grad_(True)
        leaf_weight = weight.clone().requires_grad_(True)
        fused_rms_norm.fused_rms_norm(leaf_hidden, leaf_weight, 1e-6).backward(upstream)
        grads.append((leaf_hidden.grad, leaf_weight.grad))
    assert torch.equal(grads[0][0], grads[1][0])
    assert torch.equal(grads[0][1], grads[1][1])


@pytest.mark.skipif(not _fused_engine_available(), reason="fused RMSNorm needs CUDA and Triton")
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_fused_rms_norm_handles_head_major_non_contiguous_input(dtype):
    device = _device()
    generator = torch.Generator(device=device).manual_seed(20260923)
    base = torch.randn((2, 4, 32, 128), device=device, dtype=dtype, generator=generator)
    hidden = base.transpose(1, 2)
    assert hidden.shape == (2, 32, 4, 128)
    assert hidden.stride() == (16384, 128, 4096, 1)
    assert not hidden.is_contiguous()

    weight = torch.randn((128, ), device=device, dtype=dtype, generator=generator)
    upstream = torch.randn(hidden.shape, device=device, dtype=dtype, generator=generator)

    eager_hidden = hidden.clone().detach().as_strided(hidden.shape, hidden.stride()).requires_grad_(True)
    eager_weight = weight.clone().requires_grad_(True)
    eager_out = _hf_rms_norm(eager_hidden, eager_weight, 1e-6)
    eager_out.backward(upstream)

    fused_hidden = hidden.clone().detach().as_strided(hidden.shape, hidden.stride()).requires_grad_(True)
    fused_weight = weight.clone().requires_grad_(True)
    fused_out = fused_rms_norm.fused_rms_norm(fused_hidden, fused_weight, 1e-6)
    fused_out.backward(upstream)

    _assert_ulp_close(fused_out, eager_out, max_ulp=2, min_frac_within_1=0.99, label="head-major forward")
    _assert_ulp_close(fused_hidden.grad, eager_hidden.grad, max_ulp=8, min_frac_within_1=0.95, label="head-major dx")
    _assert_ulp_close(fused_weight.grad,
                      eager_weight.grad,
                      max_ulp=8,
                      min_frac_within_1=0.95,
                      label="head-major dgamma")


@pytest.mark.skipif(not _fused_engine_available(), reason="fused RMSNorm needs CUDA and Triton")
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("scale", [1e-4, 1e4])
def test_fused_rms_norm_large_and_tiny_magnitudes(dtype, scale):
    device = _device()
    generator = torch.Generator(device=device).manual_seed(20260923)
    hidden = (scale * torch.randn((9, 2048), device=device, dtype=dtype, generator=generator)).requires_grad_(True)
    weight = torch.randn((2048, ), device=device, dtype=dtype, generator=generator).requires_grad_(True)
    upstream = torch.randn((9, 2048), device=device, dtype=dtype, generator=generator)

    eager_hidden = hidden.detach().clone().requires_grad_(True)
    eager_weight = weight.detach().clone().requires_grad_(True)
    eager_out = _hf_rms_norm(eager_hidden, eager_weight, 1e-6)
    eager_out.backward(upstream)

    fused_hidden = hidden.detach().clone().requires_grad_(True)
    fused_weight = weight.detach().clone().requires_grad_(True)
    fused_out = fused_rms_norm.fused_rms_norm(fused_hidden, fused_weight, 1e-6)
    fused_out.backward(upstream)

    _assert_ulp_close(fused_out, eager_out, max_ulp=2, min_frac_within_1=0.99, label="scaled forward")
    _assert_ulp_close(fused_hidden.grad, eager_hidden.grad, max_ulp=16, min_frac_within_1=0.90, label="scaled dx")
    _assert_ulp_close(fused_weight.grad, eager_weight.grad, max_ulp=16, min_frac_within_1=0.90, label="scaled dgamma")


@pytest.mark.skipif(not _fused_engine_available(), reason="fused RMSNorm needs CUDA and Triton")
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_fused_rms_norm_handles_strided_weight(dtype):
    device = _device()
    generator = torch.Generator(device=device).manual_seed(20260923)
    hidden = torch.randn((6, 256), device=device, dtype=dtype, generator=generator)
    # Every other element of a larger buffer: reading it as a dense array picks up the wrong values.
    weight_buffer = torch.randn((2 * 256, ), device=device, dtype=dtype, generator=generator)
    upstream = torch.randn((6, 256), device=device, dtype=dtype, generator=generator)

    eager_hidden = hidden.clone().requires_grad_(True)
    eager_weight = weight_buffer.clone()[::2].requires_grad_(True)
    eager_out = _hf_rms_norm(eager_hidden, eager_weight, 1e-6)
    eager_out.backward(upstream)

    fused_hidden = hidden.clone().requires_grad_(True)
    fused_weight = weight_buffer.clone()[::2].requires_grad_(True)
    assert fused_weight.stride() == (2, )
    fused_out = fused_rms_norm.fused_rms_norm(fused_hidden, fused_weight, 1e-6)
    fused_out.backward(upstream)

    _assert_ulp_close(fused_out, eager_out, max_ulp=2, min_frac_within_1=0.99, label="strided-weight forward")
    _assert_ulp_close(fused_hidden.grad,
                      eager_hidden.grad,
                      max_ulp=8,
                      min_frac_within_1=0.95,
                      label="strided-weight dx")
    _assert_ulp_close(fused_weight.grad,
                      eager_weight.grad,
                      max_ulp=8,
                      min_frac_within_1=0.95,
                      label="strided-weight dgamma")


@pytest.mark.skipif(not _fused_engine_available(), reason="fused RMSNorm needs CUDA and Triton")
@pytest.mark.parametrize("hidden_dtype, weight_dtype, width", [
    (torch.bfloat16, torch.bfloat16, 2049),
    (torch.float32, torch.float32, 128),
    (torch.bfloat16, torch.float16, 128),
],
                         ids=["wider-than-2048", "float32", "mixed-dtypes"])
def test_fused_rms_norm_rejects_unsupported_inputs(hidden_dtype, weight_dtype, width):
    device = _device()
    hidden = torch.randn((4, width), device=device, dtype=hidden_dtype)
    weight = torch.randn((width, ), device=device, dtype=weight_dtype)
    with pytest.raises(RuntimeError, match="fused RMSNorm"):
        fused_rms_norm.fused_rms_norm(hidden, weight, 1e-6)


@pytest.mark.skipif(not _fused_engine_available(), reason="fused RMSNorm needs CUDA and Triton")
def test_fused_rms_norm_rejects_double_backward():
    device = _device()
    generator = torch.Generator(device=device).manual_seed(20260923)
    hidden = torch.randn((4, 128), device=device, dtype=torch.bfloat16, generator=generator).requires_grad_(True)
    weight = torch.randn((128, ), device=device, dtype=torch.bfloat16, generator=generator).requires_grad_(True)
    out = fused_rms_norm.fused_rms_norm(hidden, weight, 1e-6)
    (grad_hidden, ) = torch.autograd.grad(out.float().pow(2).sum(), hidden, create_graph=True)

    # A gradient penalty needs this op's second derivative, which the kernels do not provide. It must fail rather
    # than silently leave that term out while the rest of the loss still backpropagates.
    loss = out.float().sum() + grad_hidden.float().pow(2).sum()
    with pytest.raises(RuntimeError):
        loss.backward()


def test_fused_rms_norm_fail_fast_guards_on_cpu(monkeypatch):
    monkeypatch.setattr(fused_rms_norm, "_TRITON_AVAILABLE", True)
    monkeypatch.setattr(fused_rms_norm, "_IS_ROCM_PYTORCH", False)
    hidden = torch.randn((2, 128), dtype=torch.bfloat16)
    weight = torch.randn((128, ), dtype=torch.bfloat16)
    with pytest.raises(RuntimeError, match="CUDA kernels"):
        fused_rms_norm.assert_supported(hidden, weight, 1e-6)


def test_fused_rms_norm_fail_fast_dtype_guard(monkeypatch):
    monkeypatch.setattr(fused_rms_norm, "_TRITON_AVAILABLE", True)
    monkeypatch.setattr(fused_rms_norm, "_IS_ROCM_PYTORCH", False)
    hidden = torch.randn((2, 128), dtype=torch.float32)
    weight = torch.randn((128, ), dtype=torch.float32)
    with pytest.raises(RuntimeError, match="bfloat16 and float16"):
        fused_rms_norm.assert_supported(hidden, weight, 1e-6)


@pytest.mark.parametrize("qualified_name", fused_rms_norm.SUPPORTED_RMS_NORM_CLASSES)
def test_supported_rms_norm_classes_compute_the_fused_expression(qualified_name):
    # The kernels are held to _hf_rms_norm; this holds every installable class to the same expression in the
    # installed transformers, where a class's order can change between releases (Olmo2RMSNorm's did).
    generator = torch.Generator().manual_seed(20260923)
    norm = _hf_class(qualified_name)(256).to(torch.bfloat16)
    with torch.no_grad():
        norm.weight.copy_(3 * torch.randn(256, generator=generator))
    hidden = torch.randn((64, 256), generator=generator).to(torch.bfloat16)

    with torch.no_grad():
        expected = _hf_rms_norm(hidden, norm.weight, norm.variance_epsilon)
        other_order = _gamma_before_cast_rms_norm(hidden, norm.weight, norm.variance_epsilon)
        actual = norm(hidden)
    # These inputs separate the two cast orders, so the equality below does test the order.
    assert not torch.equal(other_order, expected)
    assert actual.dtype == expected.dtype
    assert torch.equal(actual, expected)


class _GammaBeforeCastRMSNorm(torch.nn.Module):
    """Has the class-name suffix and attributes of an HF RMSNorm, but multiplies by the weight before the cast."""

    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        return _gamma_before_cast_rms_norm(hidden_states, self.weight, self.variance_epsilon)


def _gamma_before_cast_forward(self, hidden_states):
    return _gamma_before_cast_rms_norm(hidden_states, self.weight, self.variance_epsilon)


def _name_alike(monkeypatch):
    return _GammaBeforeCastRMSNorm(128)


def _hf_name_alike(qualified_name):

    def build(monkeypatch):
        return _hf_class(qualified_name)(128)

    return build


def _qwen3_moe_subclass(monkeypatch):

    class PatchedQwen3MoeRMSNorm(_hf_class(QWEN3_MOE_RMS_NORM)):
        forward = _gamma_before_cast_forward

    return PatchedQwen3MoeRMSNorm(128)


def _qwen3_moe_instance_patch(monkeypatch):
    norm = _hf_class(QWEN3_MOE_RMS_NORM)(128)
    norm.forward = types.MethodType(_gamma_before_cast_forward, norm)
    return norm


def _qwen3_moe_class_patch(monkeypatch):
    rms_norm_class = _hf_class(QWEN3_MOE_RMS_NORM)
    monkeypatch.setattr(rms_norm_class, "forward", _gamma_before_cast_forward)
    return rms_norm_class(128)


def _qwen3_moe_wrapped_class_patch(monkeypatch):
    rms_norm_class = _hf_class(QWEN3_MOE_RMS_NORM)

    # The wrapper takes the original's module and qualified name, so it cannot be told apart by name.
    @functools.wraps(rms_norm_class.forward)
    def wrapped_forward(self, hidden_states):
        return _gamma_before_cast_forward(self, hidden_states)

    monkeypatch.setattr(rms_norm_class, "forward", wrapped_forward)
    return rms_norm_class(128)


@pytest.mark.parametrize("build", [
    _name_alike,
    _hf_name_alike("transformers.models.olmo2.modeling_olmo2.Olmo2RMSNorm"),
    _hf_name_alike("transformers.models.gpt_oss.modeling_gpt_oss.GptOssRMSNorm"),
    _qwen3_moe_subclass,
    _qwen3_moe_instance_patch,
    _qwen3_moe_class_patch,
    _qwen3_moe_wrapped_class_patch,
],
                         ids=[
                             "name-alike", "olmo2", "gpt-oss", "qwen3-moe-subclass", "qwen3-moe-instance-patch",
                             "qwen3-moe-class-patch", "qwen3-moe-wrapped-class-patch"
                         ])
def test_replace_rms_norm_leaves_other_forwards_untouched(build, monkeypatch):
    generator = torch.Generator().manual_seed(20260923)
    norm = build(monkeypatch).to(torch.bfloat16)
    with torch.no_grad():
        norm.weight.copy_(3 * torch.randn(norm.weight.shape, generator=generator))
    hidden = torch.randn((16, 128), generator=generator).to(torch.bfloat16)
    with torch.no_grad():
        before = norm(hidden)

    assert fused_rms_norm.replace_rms_norm(torch.nn.Sequential(norm)) == 0
    with torch.no_grad():
        after = norm(hidden)
    assert torch.equal(after, before)


@pytest.mark.parametrize("qualified_name", fused_rms_norm.SUPPORTED_RMS_NORM_CLASSES)
def test_replace_rms_norm_runs_the_eager_forward_for_cpu_inputs(qualified_name):
    generator = torch.Generator().manual_seed(20260923)
    norm = _hf_class(qualified_name)(128).to(torch.bfloat16)
    with torch.no_grad():
        norm.weight.copy_(3 * torch.randn(128, generator=generator))
    hidden = torch.randn((16, 128), generator=generator).to(torch.bfloat16)
    with torch.no_grad():
        before = norm(hidden)

    # Replacing works before the model moves to the GPU, and until then the module computes exactly what it did.
    assert fused_rms_norm.replace_rms_norm(torch.nn.Sequential(norm)) == 1
    with torch.no_grad():
        after = norm(hidden)
    assert torch.equal(after, before)
    assert fused_rms_norm.replace_rms_norm(torch.nn.Sequential(norm)) == 0


def test_replace_rms_norm_leaves_norms_wider_than_the_kernels_alone():
    rms_norm_class = _hf_class(QWEN3_MOE_RMS_NORM)
    assert fused_rms_norm.replace_rms_norm(rms_norm_class(2048)) == 1
    assert fused_rms_norm.replace_rms_norm(rms_norm_class(2049)) == 0


@pytest.mark.skipif(not _fused_engine_available(), reason="fused RMSNorm needs CUDA and Triton")
@pytest.mark.parametrize("qualified_name", fused_rms_norm.SUPPORTED_RMS_NORM_CLASSES)
def test_replace_rms_norm_fuses_supported_norms(qualified_name):
    rms_norm_class = _hf_class(qualified_name)
    device = _device()
    generator = torch.Generator(device=device).manual_seed(20260923)
    # A hidden-size norm and a head-dim norm. The hidden norm's epsilon moves its output by many ULPs, so a forward
    # that ignored the module's own epsilon would fail.
    eager = torch.nn.ModuleDict({
        "hidden_norm": rms_norm_class(256, eps=1e-2),
        "head_norm": rms_norm_class(128, eps=1e-6),
    }).to(device=device, dtype=torch.bfloat16)
    with torch.no_grad():
        for norm in eager.values():
            norm.weight.copy_(torch.randn(norm.weight.shape, device=device, generator=generator))
    fused = copy.deepcopy(eager)
    assert fused_rms_norm.replace_rms_norm(fused) == 2
    assert fused_rms_norm.replace_rms_norm(fused) == 0

    hidden = 0.1 * torch.randn((2, 8, 256), device=device, dtype=torch.bfloat16, generator=generator)
    heads = torch.randn((2, 8, 4, 128), device=device, dtype=torch.bfloat16, generator=generator)
    hidden_upstream = torch.randn((2, 8, 256), device=device, dtype=torch.bfloat16, generator=generator)
    heads_upstream = torch.randn((2, 4, 8, 128), device=device, dtype=torch.bfloat16, generator=generator)

    def run(norms):
        hidden_in = hidden.clone().requires_grad_(True)
        heads_in = heads.clone().requires_grad_(True)
        hidden_out = norms["hidden_norm"](hidden_in)
        # Attention transposes q and k after their norm, so the gradient reaching the head norm is head-major.
        heads_out = norms["head_norm"](heads_in).transpose(1, 2)
        torch.autograd.backward((hidden_out, heads_out), (hidden_upstream, heads_upstream))
        return {
            "hidden forward": hidden_out,
            "head forward": heads_out,
            "hidden dx": hidden_in.grad,
            "head dx": heads_in.grad,
            "hidden dgamma": norms["hidden_norm"].weight.grad,
            "head dgamma": norms["head_norm"].weight.grad,
        }

    eager_results = run(eager)
    fused_results = run(fused)
    for label in ("hidden forward", "head forward"):
        _assert_ulp_close(fused_results[label], eager_results[label], max_ulp=2, min_frac_within_1=0.99, label=label)
    for label in ("hidden dx", "head dx", "hidden dgamma", "head dgamma"):
        _assert_ulp_close(fused_results[label], eager_results[label], max_ulp=8, min_frac_within_1=0.95, label=label)


@pytest.mark.skipif(not _fused_engine_available(), reason="fused RMSNorm needs CUDA and Triton")
def test_replace_rms_norm_runs_the_kernels_only_where_they_apply():
    device = _device()
    generator = torch.Generator(device=device).manual_seed(20260923)
    norm = _hf_class(QWEN3_MOE_RMS_NORM)(128).to(device=device, dtype=torch.bfloat16)
    with torch.no_grad():
        norm.weight.copy_(torch.randn(128, device=device, generator=generator))
    # Head-major, as attention lays out q and k. Eager keeps that layout in its output while the kernels return a
    # contiguous one, so the output's layout shows which path a call took.
    heads = torch.randn((2, 4, 8, 128), device=device, dtype=torch.bfloat16, generator=generator).transpose(1, 2)
    with torch.no_grad():
        eager_out = norm(heads)
        eager_float_out = norm(heads.float())
        kernel_out = fused_rms_norm.fused_rms_norm(heads, norm.weight, norm.variance_epsilon)
    assert not eager_out.is_contiguous()

    assert fused_rms_norm.replace_rms_norm(torch.nn.Sequential(norm)) == 1
    with torch.no_grad():
        fused_out = norm(heads)
        # The kernels take no float32 input, so this one runs the class's eager forward and gets exactly its result.
        fallback_out = norm(heads.float())
    assert fused_out.is_contiguous()
    assert torch.equal(fused_out, kernel_out)
    assert torch.equal(fallback_out, eager_float_out)
