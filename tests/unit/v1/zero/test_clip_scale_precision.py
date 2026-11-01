# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team

from types import SimpleNamespace

import pytest
import torch
import deepspeed.runtime.utils as ds_utils
from deepspeed.accelerator import get_accelerator
from deepspeed.runtime.zero.stage3 import DeepSpeedZeroOptimizer_Stage3
from deepspeed.runtime.zero.stage_1_and_2 import DeepSpeedZeroOptimizer

# Rounded to bf16, the clip coefficient for this norm (clip_grad=1.0) changes about a quarter of bf16 gradients.
TOTAL_NORM = 1.0033


def _device():
    return get_accelerator().device_name()


def _grad(dtype):
    generator = torch.Generator().manual_seed(0)
    return torch.randn(4096, generator=generator).to(device=_device(), dtype=dtype)


def _clipped(grad, total_norm):
    clip = torch.clamp(((total_norm / 1.0) + 1e-6) / 1.0, min=1.0) * 1.0
    return (grad.float() * (1. / clip)).to(grad.dtype)


def _optimizer(cls):
    optimizer = object.__new__(cls)
    optimizer.clip_grad = 1.0
    optimizer.custom_loss_scaler = False
    optimizer.loss_scaler = SimpleNamespace(cur_scale=1.0)
    optimizer.device = _device()
    return optimizer


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_scale_grads_applies_tensor_scale_at_full_precision(dtype, monkeypatch):
    monkeypatch.setattr(ds_utils, "_SCALE_CHUNK_NUMEL", 1000)
    scale = torch.tensor(1 / TOTAL_NORM, device=_device())
    grad = _grad(dtype)
    expected = (grad.float() * scale).to(dtype)
    ds_utils.scale_grads_(grad, scale)
    assert torch.equal(grad, expected)


def test_scale_grads_keeps_plain_mul_for_full_precision_grads():
    scale = torch.tensor(1 / TOTAL_NORM, device=_device())
    grad = _grad(torch.float32)
    expected = grad * scale
    ds_utils.scale_grads_(grad, scale)
    assert torch.equal(grad, expected)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_zero_1_2_clips_low_precision_grads_with_full_precision_coefficient(dtype):
    total_norm = torch.tensor(TOTAL_NORM, device=_device())
    grad, sub_partition = _grad(dtype), _grad(dtype)
    expected = _clipped(grad, total_norm)
    _optimizer(DeepSpeedZeroOptimizer).unscale_and_clip_grads([grad, [sub_partition]], total_norm)
    assert torch.equal(grad, expected)
    assert torch.equal(sub_partition, expected)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_zero_3_clips_low_precision_grads_with_full_precision_coefficient(dtype):
    total_norm = torch.tensor(TOTAL_NORM, device=_device())
    optimizer = _optimizer(DeepSpeedZeroOptimizer_Stage3)
    partition = SimpleNamespace(grad=_grad(dtype))
    optimizer.fp32_partitioned_groups_flat = [partition]
    expected = _clipped(partition.grad, total_norm)
    optimizer.unscale_and_clip_grads(0, total_norm)
    assert torch.equal(partition.grad, expected)
