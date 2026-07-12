# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import pytest
import torch

try:
    from deepspeed.ops.adam.cpu_adam import DeepSpeedCPUAdam, _malloc_trim
    _cpu_adam_available = True
except (ImportError, ModuleNotFoundError):
    _cpu_adam_available = False

try:
    from deepspeed.runtime.zero.stage_1_and_2 import DeepSpeedZeroOptimizer
    _zero_optimizer_available = True
except (ImportError, ModuleNotFoundError):
    _zero_optimizer_available = False


@pytest.mark.skipif(not _cpu_adam_available, reason="DeepSpeedCPUAdam not available")
def test_malloc_trim_helper_exists():
    """Verify _malloc_trim helper is importable and callable."""
    assert callable(_malloc_trim)
    # Should not raise on any platform (no-op on non-Linux)
    _malloc_trim()


@pytest.mark.skipif(not _cpu_adam_available, reason="DeepSpeedCPUAdam not available")
def test_malloc_trim_called_per_param_group(monkeypatch):
    """Verify _malloc_trim runs once per param group on first step.

    ZeRO-2 CPU offload invokes step() once per group; an optimizer-wide flag
    would skip trim for later groups (Codex review on #8132).
    """
    p0 = torch.nn.Parameter(torch.randn(50, device='cpu'))
    p1 = torch.nn.Parameter(torch.randn(50, device='cpu'))
    optimizer = DeepSpeedCPUAdam([{'params': [p0]}, {'params': [p1]}], lr=1e-3)

    call_count = 0

    def fake_trim():
        nonlocal call_count
        call_count += 1

    monkeypatch.setattr("deepspeed.ops.adam.cpu_adam._malloc_trim", fake_trim)

    # First call: both groups step together (normal DeepSpeedCPUAdam.step)
    p0.grad = torch.randn_like(p0)
    p1.grad = torch.randn_like(p1)
    optimizer.step()
    assert call_count == 2, "malloc_trim should run once per param group"

    # Second call: no additional trim
    p0.grad = torch.randn_like(p0)
    p1.grad = torch.randn_like(p1)
    optimizer.step()
    assert call_count == 2, "malloc_trim should not re-run for already-trimmed groups"


@pytest.mark.skipif(not _cpu_adam_available, reason="DeepSpeedCPUAdam not available")
def test_cpu_adam_first_step_initializes_optimizer_states():
    """Smoke test: first CPUAdam step creates contiguous fp32 optimizer states."""
    param = torch.randn(1000, device='cpu', dtype=torch.float32)
    grad = torch.randn(1000, device='cpu', dtype=torch.float32)
    optimizer = DeepSpeedCPUAdam([param], lr=1e-3)

    param.grad = grad
    optimizer.step()

    state = optimizer.state[param]
    assert 'exp_avg' in state
    assert 'exp_avg_sq' in state
    assert state['exp_avg'].is_contiguous()
    assert state['exp_avg_sq'].is_contiguous()
    assert state['exp_avg'].numel() == param.numel()
    assert state['exp_avg_sq'].numel() == param.numel()


@pytest.mark.skipif(not _zero_optimizer_available, reason="DeepSpeedZeroOptimizer not available")
def test_unscale_and_clip_grads_cpu_offload_uses_float_scale():
    """On cpu_offload, combined_scale is converted to float before mul_."""

    class MockOptimizer:
        loss_scale = 16.0
        clip_grad = 1.0
        cpu_offload = True

        unscale_and_clip_grads = DeepSpeedZeroOptimizer.unscale_and_clip_grads

    opt = MockOptimizer()
    grad = torch.randn(100, device='cpu', dtype=torch.float32)
    original_data_ptr = grad.data_ptr()
    total_norm = torch.tensor(10.0, device='cpu', dtype=torch.float32)

    opt.unscale_and_clip_grads([grad], total_norm)

    assert grad.data_ptr() == original_data_ptr, \
        "unscale_and_clip_grads should not allocate a new tensor"


@pytest.mark.skipif(not _zero_optimizer_available, reason="DeepSpeedZeroOptimizer not available")
def test_unscale_and_clip_grads_no_clip():
    """Verify unscale_and_clip_grads works correctly when clip_grad is 0."""

    class MockOptimizer:
        loss_scale = 16.0
        clip_grad = 0.0
        cpu_offload = True

        unscale_and_clip_grads = DeepSpeedZeroOptimizer.unscale_and_clip_grads

    opt = MockOptimizer()
    grad = torch.randn(100, device='cpu', dtype=torch.float32)
    original_data_ptr = grad.data_ptr()
    original_values = grad.clone()

    total_norm = torch.tensor(10.0, device='cpu', dtype=torch.float32)
    opt.unscale_and_clip_grads([grad], total_norm)

    expected = original_values / opt.loss_scale
    assert torch.allclose(grad, expected, rtol=1e-5)
    assert grad.data_ptr() == original_data_ptr


@pytest.mark.skipif(not _zero_optimizer_available, reason="DeepSpeedZeroOptimizer not available")
def test_unscale_and_clip_grads_gpu_path_skips_float_conversion():
    """Without cpu_offload, do not force float() (avoids GPU sync regression)."""

    class MockOptimizer:
        loss_scale = 16.0
        clip_grad = 0.0
        cpu_offload = False

        unscale_and_clip_grads = DeepSpeedZeroOptimizer.unscale_and_clip_grads

    opt = MockOptimizer()
    grad = torch.randn(100, device='cpu', dtype=torch.float32)
    original = grad.clone()
    total_norm = torch.tensor(10.0, device='cpu', dtype=torch.float32)

    # clip_grad==0 keeps combined_scale as loss_scale (float already); just
    # ensure the non-offload path still scales correctly.
    opt.unscale_and_clip_grads([grad], total_norm)
    assert torch.allclose(grad, original / opt.loss_scale, rtol=1e-5)
