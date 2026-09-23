# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team
"""Compare fused RMSNorm with the HF eager expression."""

import pytest
import torch

from deepspeed.accelerator import get_accelerator
from deepspeed.ops.triton_ops import fused_rms_norm


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
@pytest.mark.parametrize("shape", [(0, 2048), (7, 128), (5, 2048), (3, 130)])
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
    upstream = torch.randn_like(hidden)

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
    weight = torch.randn((128, ), dtype=torch.float32)
    with pytest.raises(RuntimeError, match="bfloat16 and float16"):
        fused_rms_norm._assert_supported_device_and_dtype(weight, name="weight")


def test_replace_rms_norm_matches_only_hf_contract(monkeypatch):
    monkeypatch.setattr(fused_rms_norm, "_TRITON_AVAILABLE", True)
    monkeypatch.setattr(fused_rms_norm, "_IS_ROCM_PYTORCH", False)
    checked = []

    def fake_assert(tensor, *, name):
        checked.append((name, tuple(tensor.shape)))

    monkeypatch.setattr(fused_rms_norm, "_assert_supported_device_and_dtype", fake_assert)

    class Qwen3MoeRMSNorm(torch.nn.Module):

        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(128, dtype=torch.bfloat16))
            self.variance_epsilon = 1e-6

        def forward(self, hidden_states):
            return hidden_states

    class PlainLayerNorm(torch.nn.Module):

        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(128, dtype=torch.bfloat16))
            self.variance_epsilon = 1e-6

    model = torch.nn.Sequential(Qwen3MoeRMSNorm(), PlainLayerNorm())
    assert fused_rms_norm.replace_rms_norm(model) == 1
    assert checked == [("Qwen3MoeRMSNorm.weight", (128, ))]
