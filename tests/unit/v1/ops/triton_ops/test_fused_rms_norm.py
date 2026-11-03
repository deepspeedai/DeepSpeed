# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team
"""Compare fused RMSNorm with the HF eager expression."""

import copy
import functools
import importlib
import io
import os
from pathlib import Path
import subprocess
import sys
import types

import pytest
import torch

from deepspeed.accelerator import get_accelerator
from unit.common import DistributedTest

QWEN3_MOE_RMS_NORM = "transformers.models.qwen3_moe.modeling_qwen3_moe.Qwen3MoeRMSNorm"
# Keep parametrization collection-safe: importing the Triton ops here would probe CUDA before --forked.
RMS_NORM_CLASSES = (
    "transformers.models.deepseek_v2.modeling_deepseek_v2.DeepseekV2RMSNorm",
    "transformers.models.deepseek_v3.modeling_deepseek_v3.DeepseekV3RMSNorm",
    "transformers.models.llama.modeling_llama.LlamaRMSNorm",
    "transformers.models.mistral.modeling_mistral.MistralRMSNorm",
    "transformers.models.mixtral.modeling_mixtral.MixtralRMSNorm",
    "transformers.models.phi3.modeling_phi3.Phi3RMSNorm",
    "transformers.models.qwen2.modeling_qwen2.Qwen2RMSNorm",
    "transformers.models.qwen2_moe.modeling_qwen2_moe.Qwen2MoeRMSNorm",
    "transformers.models.qwen3.modeling_qwen3.Qwen3RMSNorm",
    QWEN3_MOE_RMS_NORM,
)


def _rms_norm():
    from deepspeed.ops.triton_ops import fused_rms_norm
    return fused_rms_norm


def _fused_engine_available():
    accelerator = get_accelerator()
    return (accelerator.is_available() and accelerator.device_name().startswith("cuda") and _rms_norm().is_available())


def _require_fused_engine():
    if not _fused_engine_available():
        pytest.skip("fused RMSNorm needs CUDA and Triton")


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


def _check_fused_rms_norm_matches_hf_forward_and_backward(dtype, shape):
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
    fused_out = _rms_norm().fused_rms_norm(fused_hidden, fused_weight, 1e-6)
    fused_out.backward(upstream)

    _assert_ulp_close(fused_out, eager_out, max_ulp=2, min_frac_within_1=0.99, label="forward")
    _assert_ulp_close(fused_hidden.grad, eager_hidden.grad, max_ulp=8, min_frac_within_1=0.95, label="dx")
    _assert_ulp_close(fused_weight.grad, eager_weight.grad, max_ulp=8, min_frac_within_1=0.95, label="dgamma")


def _check_fused_rms_norm_handles_head_major_non_contiguous_input(dtype):
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
    fused_out = _rms_norm().fused_rms_norm(fused_hidden, fused_weight, 1e-6)
    fused_out.backward(upstream)

    _assert_ulp_close(fused_out, eager_out, max_ulp=2, min_frac_within_1=0.99, label="head-major forward")
    _assert_ulp_close(fused_hidden.grad, eager_hidden.grad, max_ulp=8, min_frac_within_1=0.95, label="head-major dx")
    _assert_ulp_close(fused_weight.grad,
                      eager_weight.grad,
                      max_ulp=8,
                      min_frac_within_1=0.95,
                      label="head-major dgamma")


def _check_fused_rms_norm_large_and_tiny_magnitudes(dtype, scale):
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
    fused_out = _rms_norm().fused_rms_norm(fused_hidden, fused_weight, 1e-6)
    fused_out.backward(upstream)

    _assert_ulp_close(fused_out, eager_out, max_ulp=2, min_frac_within_1=0.99, label="scaled forward")
    _assert_ulp_close(fused_hidden.grad, eager_hidden.grad, max_ulp=16, min_frac_within_1=0.90, label="scaled dx")
    _assert_ulp_close(fused_weight.grad, eager_weight.grad, max_ulp=16, min_frac_within_1=0.90, label="scaled dgamma")


def _check_fused_rms_norm_handles_strided_weight(dtype):
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
    fused_out = _rms_norm().fused_rms_norm(fused_hidden, fused_weight, 1e-6)
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


def _check_fused_rms_norm_rejects_unsupported_inputs(hidden_dtype, weight_dtype, width):
    device = _device()
    hidden = torch.randn((4, width), device=device, dtype=hidden_dtype)
    weight = torch.randn((width, ), device=device, dtype=weight_dtype)
    with pytest.raises(RuntimeError, match="fused RMSNorm"):
        _rms_norm().fused_rms_norm(hidden, weight, 1e-6)


def _check_fused_rms_norm_rejects_double_backward():
    device = _device()
    generator = torch.Generator(device=device).manual_seed(20260923)
    hidden = torch.randn((4, 128), device=device, dtype=torch.bfloat16, generator=generator).requires_grad_(True)
    weight = torch.randn((128, ), device=device, dtype=torch.bfloat16, generator=generator).requires_grad_(True)
    out = _rms_norm().fused_rms_norm(hidden, weight, 1e-6)
    (grad_hidden, ) = torch.autograd.grad(out.float().pow(2).sum(), hidden, create_graph=True)

    # A gradient penalty needs this op's second derivative, which the kernels do not provide. It must fail rather
    # than silently leave that term out while the rest of the loss still backpropagates.
    loss = out.float().sum() + grad_hidden.float().pow(2).sum()
    with pytest.raises(RuntimeError):
        loss.backward()


def test_fused_rms_norm_fail_fast_guards_on_cpu(monkeypatch):
    fused_rms_norm = _rms_norm()
    monkeypatch.setattr(fused_rms_norm, "_TRITON_AVAILABLE", True)
    monkeypatch.setattr(fused_rms_norm, "_IS_ROCM_PYTORCH", False)
    hidden = torch.randn((2, 128), dtype=torch.bfloat16)
    weight = torch.randn((128, ), dtype=torch.bfloat16)
    with pytest.raises(RuntimeError, match="CUDA kernels"):
        fused_rms_norm.assert_supported(hidden, weight, 1e-6)


def test_fused_rms_norm_fail_fast_dtype_guard(monkeypatch):
    fused_rms_norm = _rms_norm()
    monkeypatch.setattr(fused_rms_norm, "_TRITON_AVAILABLE", True)
    monkeypatch.setattr(fused_rms_norm, "_IS_ROCM_PYTORCH", False)
    hidden = torch.randn((2, 128), dtype=torch.float32)
    weight = torch.randn((128, ), dtype=torch.float32)
    with pytest.raises(RuntimeError, match="bfloat16 and float16"):
        fused_rms_norm.assert_supported(hidden, weight, 1e-6)


@pytest.mark.parametrize("qualified_name", RMS_NORM_CLASSES)
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

    assert _rms_norm().replace_rms_norm(torch.nn.Sequential(norm)) == 0
    with torch.no_grad():
        after = norm(hidden)
    assert torch.equal(after, before)


@pytest.mark.parametrize("qualified_name", RMS_NORM_CLASSES)
def test_replace_rms_norm_runs_the_eager_forward_for_cpu_inputs(qualified_name):
    generator = torch.Generator().manual_seed(20260923)
    norm = _hf_class(qualified_name)(128).to(torch.bfloat16)
    with torch.no_grad():
        norm.weight.copy_(3 * torch.randn(128, generator=generator))
    hidden = torch.randn((16, 128), generator=generator).to(torch.bfloat16)
    with torch.no_grad():
        before = norm(hidden)

    # Replacing works before the model moves to the GPU, and until then the module computes exactly what it did.
    assert _rms_norm().replace_rms_norm(torch.nn.Sequential(norm)) == 1
    with torch.no_grad():
        after = norm(hidden)
    assert torch.equal(after, before)
    assert _rms_norm().replace_rms_norm(torch.nn.Sequential(norm)) == 0


def test_replace_rms_norm_leaves_norms_wider_than_the_kernels_alone():
    rms_norm_class = _hf_class(QWEN3_MOE_RMS_NORM)
    assert _rms_norm().replace_rms_norm(rms_norm_class(2048)) == 1
    assert _rms_norm().replace_rms_norm(rms_norm_class(2049)) == 0


def _check_replace_rms_norm_copy_uses_its_own_epsilon(device, copy_module):
    if device == "cuda":
        if not _fused_engine_available():
            pytest.skip("fused RMSNorm needs CUDA and Triton")
        device = _device()
    norm = _hf_class(QWEN3_MOE_RMS_NORM)(128).to(device=device, dtype=torch.bfloat16)
    hidden = torch.full((4, 128), 0.25, device=device, dtype=torch.bfloat16)
    with torch.no_grad():
        original_output = norm(hidden)
    assert _rms_norm().replace_rms_norm(norm) == 1

    replica = copy_module(norm)
    assert _rms_norm().replace_rms_norm(replica) == 0
    replica.variance_epsilon = 0.25
    with torch.no_grad():
        expected = _hf_rms_norm(hidden, replica.weight, replica.variance_epsilon)
        actual = replica(hidden)
        original_after = norm(hidden)
    assert not torch.equal(expected, original_output)
    assert torch.equal(original_after, original_output)
    if hidden.device.type == "cpu":
        assert torch.equal(actual, expected)
    else:
        _assert_ulp_close(actual, expected, max_ulp=2, min_frac_within_1=0.99, label="shallow-copy forward")


def _check_replace_rms_norm_model_round_trip_preserves_outputs_and_gradients(device):
    if device == "cuda":
        if not _fused_engine_available():
            pytest.skip("fused RMSNorm needs CUDA and Triton")
        device = _device()
    norm = _hf_class(QWEN3_MOE_RMS_NORM)(128, eps=0.25)
    model = torch.nn.Sequential(norm).to(device=device, dtype=torch.bfloat16)
    state_dict_keys = tuple(model.state_dict())
    assert _rms_norm().replace_rms_norm(model) == 1

    serialized = io.BytesIO()
    torch.save(model, serialized)
    serialized.seek(0)
    loaded = torch.load(serialized, weights_only=False)
    assert tuple(loaded.state_dict()) == state_dict_keys
    assert torch.equal(loaded[0].weight, model[0].weight)
    hidden = torch.full((2, 4, 8, 128), 0.5, device=device, dtype=torch.bfloat16).transpose(1, 2)
    with torch.no_grad():
        before_reinstall = loaded(hidden)
        eager_before_reinstall = _hf_rms_norm(hidden, loaded[0].weight, loaded[0].variance_epsilon)
    assert torch.equal(before_reinstall, eager_before_reinstall)
    assert before_reinstall.stride() == eager_before_reinstall.stride()
    assert _rms_norm().replace_rms_norm(loaded) == 1
    assert _rms_norm().replace_rms_norm(loaded) == 0
    eager_hidden = hidden.clone().requires_grad_(True)
    eager_weight = model[0].weight.detach().clone().requires_grad_(True)
    expected = _hf_rms_norm(eager_hidden, eager_weight, norm.variance_epsilon)
    expected.float().sum().backward()
    loaded_hidden = hidden.clone().requires_grad_(True)
    actual = loaded(loaded_hidden)
    actual.float().sum().backward()

    if hidden.device.type == "cpu":
        assert torch.equal(actual, expected)
        assert torch.equal(loaded_hidden.grad, eager_hidden.grad)
        assert torch.equal(loaded[0].weight.grad, eager_weight.grad)
    else:
        _assert_ulp_close(actual, expected, max_ulp=2, min_frac_within_1=0.99, label="loaded forward")
        _assert_ulp_close(loaded_hidden.grad, eager_hidden.grad, max_ulp=8, min_frac_within_1=0.95, label="loaded dx")
        _assert_ulp_close(loaded[0].weight.grad,
                          eager_weight.grad,
                          max_ulp=8,
                          min_frac_within_1=0.95,
                          label="loaded dgamma")
    assert model[0].weight.grad is None


@pytest.mark.parametrize("dispatcher_installed", [False, True], ids=["no-dispatcher", "installed-dispatcher"])
def test_replace_rms_norm_checkpoint_loads_in_a_fresh_process(tmp_path, dispatcher_installed):
    norm = _hf_class(QWEN3_MOE_RMS_NORM)(128, eps=0.25).to(torch.bfloat16)
    model = torch.nn.Sequential(norm)
    hidden = torch.full((4, 128), 0.5, dtype=torch.bfloat16)
    with torch.no_grad():
        expected = _hf_rms_norm(hidden, norm.weight, norm.variance_epsilon)
    assert _rms_norm().replace_rms_norm(model) == 1
    checkpoint = tmp_path / "rms_norm.pt"
    torch.save({"model": model, "hidden": hidden, "expected": expected}, checkpoint)

    # Loading must not depend on a class dispatcher left installed in the saving process.
    script = """
import sys
import torch

if sys.argv[2] == "installed":
    from deepspeed.ops.triton_ops.fused_rms_norm import replace_rms_norm
    from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeRMSNorm
    assert replace_rms_norm(Qwen3MoeRMSNorm(128)) == 1

checkpoint = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
model = checkpoint["model"]
hidden = checkpoint["hidden"]
expected = checkpoint["expected"]
assert torch.equal(model(hidden), expected)

from deepspeed.ops.triton_ops.fused_rms_norm import replace_rms_norm
assert replace_rms_norm(model) == 1
assert replace_rms_norm(model) == 0
assert torch.equal(model(hidden), expected)
"""
    source_root = Path(_rms_norm().__file__).resolve().parents[3]
    mode = "installed" if dispatcher_installed else "not-installed"
    result = subprocess.run([sys.executable, "-c", script, str(checkpoint), mode],
                            cwd=source_root,
                            capture_output=True,
                            text=True,
                            timeout=120)
    assert result.returncode == 0, result.stdout + result.stderr


def _check_replace_rms_norm_data_parallel_matches_eager_training():
    if get_accelerator().device_count() < 2:
        pytest.skip("DataParallel replication needs two CUDA devices")
    device = get_accelerator().device_name(0)
    model = torch.nn.Sequential(_hf_class(QWEN3_MOE_RMS_NORM)(128, eps=0.25))
    model = model.to(device=device, dtype=torch.bfloat16)
    eager_weight = torch.nn.Parameter(model[0].weight.detach().clone())
    optimizer = torch.optim.SGD(model.parameters(), lr=1 / 32)
    eager_optimizer = torch.optim.SGD([eager_weight], lr=1 / 32)
    assert _rms_norm().replace_rms_norm(model) == 1
    parallel = torch.nn.DataParallel(model, device_ids=[0, 1])
    # Constant rows keep DataParallel's extra BF16 gradient reduction exact.
    hidden = torch.full((8, 128), 0.5, device=device, dtype=torch.bfloat16)
    for _ in range(2):
        actual_hidden = hidden.clone().requires_grad_(True)
        eager_hidden = hidden.clone().requires_grad_(True)
        actual = parallel(actual_hidden)
        expected = _hf_rms_norm(eager_hidden, eager_weight, model[0].variance_epsilon)
        actual.float().sum().backward()
        expected.float().sum().backward()
        _assert_ulp_close(actual, expected, max_ulp=2, min_frac_within_1=0.99, label="DataParallel forward")
        _assert_ulp_close(actual_hidden.grad,
                          eager_hidden.grad,
                          max_ulp=8,
                          min_frac_within_1=0.95,
                          label="DataParallel dx")
        _assert_ulp_close(model[0].weight.grad,
                          eager_weight.grad,
                          max_ulp=8,
                          min_frac_within_1=0.95,
                          label="DataParallel dgamma")
        optimizer.step()
        eager_optimizer.step()
        assert torch.equal(model[0].weight, eager_weight)
        optimizer.zero_grad()
        eager_optimizer.zero_grad()


def _check_replace_rms_norm_fuses_supported_norms(qualified_name):
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
    assert _rms_norm().replace_rms_norm(fused) == 2
    assert _rms_norm().replace_rms_norm(fused) == 0

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


def _check_replace_rms_norm_runs_the_kernels_only_where_they_apply():
    device = _device()
    generator = torch.Generator(device=device).manual_seed(20260923)
    norm = _hf_class(QWEN3_MOE_RMS_NORM)(128).to(device=device, dtype=torch.bfloat16)
    with torch.no_grad():
        norm.weight.copy_(torch.randn(128, device=device, generator=generator))
    untouched = copy.deepcopy(norm)
    # Head-major, as attention lays out q and k. Eager keeps that layout in its output while the kernels return a
    # contiguous one, so the output's layout shows which path a call took.
    heads = torch.randn((2, 4, 8, 128), device=device, dtype=torch.bfloat16, generator=generator).transpose(1, 2)
    with torch.no_grad():
        eager_out = norm(heads)
        eager_float_out = norm(heads.float())
        kernel_out = _rms_norm().fused_rms_norm(heads, norm.weight, norm.variance_epsilon)
    assert not eager_out.is_contiguous()

    assert _rms_norm().replace_rms_norm(torch.nn.Sequential(norm)) == 1
    with torch.no_grad():
        fused_out = norm(heads)
        # The kernels take no float32 input, so this one runs the class's eager forward and gets exactly its result.
        fallback_out = norm(heads.float())
        untouched_out = untouched(heads)
    assert fused_out.is_contiguous()
    assert torch.equal(fused_out, kernel_out)
    assert torch.equal(fallback_out, eager_float_out)
    assert torch.equal(untouched_out, eager_out)
    assert not untouched_out.is_contiguous()
    assert _rms_norm().replace_rms_norm(untouched) == 1
    assert _rms_norm().replace_rms_norm(untouched) == 0
    with torch.no_grad():
        assert torch.equal(untouched(heads), kernel_out)


@pytest.mark.parametrize("copy_module", [copy.copy, copy.deepcopy], ids=["shallow-copy", "deep-copy"])
def test_replace_rms_norm_copy_uses_its_own_epsilon(copy_module):
    _check_replace_rms_norm_copy_uses_its_own_epsilon("cpu", copy_module)


def test_replace_rms_norm_model_round_trip_preserves_outputs_and_gradients():
    _check_replace_rms_norm_model_round_trip_preserves_outputs_and_gradients("cpu")


@pytest.mark.skipif(os.environ.get("DS_ACCELERATOR") == "cpu", reason="fused RMSNorm needs CUDA and Triton")
class TestFusedRMSNormCUDA(DistributedTest):
    # Fresh workers also tolerate CUDA state created while other test modules are being collected.
    world_size = 1
    init_distributed = False

    @pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
    @pytest.mark.parametrize("shape", [(0, 2048), (7, 128), (5, 2048), (3, 130)])
    def test_fused_rms_norm_matches_hf_forward_and_backward(self, dtype, shape):
        _require_fused_engine()
        _check_fused_rms_norm_matches_hf_forward_and_backward(dtype, shape)

    @pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
    def test_fused_rms_norm_handles_head_major_non_contiguous_input(self, dtype):
        _require_fused_engine()
        _check_fused_rms_norm_handles_head_major_non_contiguous_input(dtype)

    @pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
    @pytest.mark.parametrize("scale", [1e-4, 1e4])
    def test_fused_rms_norm_large_and_tiny_magnitudes(self, dtype, scale):
        _require_fused_engine()
        _check_fused_rms_norm_large_and_tiny_magnitudes(dtype, scale)

    @pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
    def test_fused_rms_norm_handles_strided_weight(self, dtype):
        _require_fused_engine()
        _check_fused_rms_norm_handles_strided_weight(dtype)

    @pytest.mark.parametrize("hidden_dtype, weight_dtype, width", [
        (torch.bfloat16, torch.bfloat16, 2049),
        (torch.float32, torch.float32, 128),
        (torch.bfloat16, torch.float16, 128),
    ],
                             ids=["wider-than-2048", "float32", "mixed-dtypes"])
    def test_fused_rms_norm_rejects_unsupported_inputs(self, hidden_dtype, weight_dtype, width):
        _require_fused_engine()
        _check_fused_rms_norm_rejects_unsupported_inputs(hidden_dtype, weight_dtype, width)

    def test_fused_rms_norm_rejects_double_backward(self):
        _require_fused_engine()
        _check_fused_rms_norm_rejects_double_backward()

    @pytest.mark.parametrize("copy_module", [copy.copy, copy.deepcopy], ids=["shallow-copy", "deep-copy"])
    def test_replace_rms_norm_copy_uses_its_own_epsilon(self, copy_module):
        _require_fused_engine()
        _check_replace_rms_norm_copy_uses_its_own_epsilon("cuda", copy_module)

    def test_replace_rms_norm_model_round_trip_preserves_outputs_and_gradients(self):
        _require_fused_engine()
        _check_replace_rms_norm_model_round_trip_preserves_outputs_and_gradients("cuda")

    def test_replace_rms_norm_data_parallel_matches_eager_training(self):
        _require_fused_engine()
        _check_replace_rms_norm_data_parallel_matches_eager_training()

    @pytest.mark.parametrize("qualified_name", RMS_NORM_CLASSES)
    def test_replace_rms_norm_fuses_supported_norms(self, qualified_name):
        _require_fused_engine()
        _check_replace_rms_norm_fuses_supported_norms(qualified_name)

    def test_replace_rms_norm_runs_the_kernels_only_where_they_apply(self):
        _require_fused_engine()
        _check_replace_rms_norm_runs_the_kernels_only_where_they_apply()
