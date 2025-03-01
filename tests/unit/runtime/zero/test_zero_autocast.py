# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

from copy import deepcopy

import pytest

import torch
from torch.nn.parallel import DistributedDataParallel as DDP

from unit.common import DistributedTest, enable_determinism
from unit.simple_model import SimpleModel
from unit.util import bf16_required_version_check

import deepspeed
from deepspeed.accelerator import get_accelerator
from deepspeed.runtime.zero import GatheredParameters

RTOL = 0.1
ATOL = 0.0


def step_amp(enabled, baseline_model, baseline_optimizer, target_engine, dtype, baseline_scaler, x, y, rtol, atol):
    # Runs the forward pass with autocasting.
    with torch.autocast(device_type="cuda", dtype=dtype, enabled=enabled):
        baseline_optimizer.zero_grad()
        baseline_loss = baseline_model(x, y)

    baseline_scaler.scale(baseline_loss).backward()
    baseline_scaler.step(baseline_optimizer)
    baseline_scaler.update()

    target_loss = target_engine(x, y)

    assert torch.allclose(baseline_loss.float(), target_loss.float(), rtol=rtol, atol=atol)

    target_engine.backward(target_loss)
    target_engine.step()


@enable_determinism(123)
def compare_loss(enable, zero_stage, dtype, autocast_conf):
    iteration = 5
    hidden_dim = 10
    lr = 0.001

    if dtype == torch.bfloat16 and not bf16_required_version_check():
        raise ValueError(
            "DeepSpeed BFloat16 tests need torch >= 1.10, NCCL >= 2.10.3, CUDA > =11.0 and HW support for BFloat16 to run correctly"
        )

    config_dict = {
        "train_micro_batch_size_per_gpu": 1,
        "steps_per_print": 1,
        "zero_optimization": {
            "stage": zero_stage,
        },
        "torch_autocast": autocast_conf,
    }

    model_cls = SimpleModel
    model = model_cls(hidden_dim)

    deepspeed.init_distributed(dist_backend='nccl')

    i = get_accelerator().current_device()
    device = get_accelerator().current_device_name()
    baseline_model = DDP(deepcopy(model).to(device=device, dtype=torch.float32), device_ids=[i], output_device=i)
    baseline_optimizer = torch.optim.AdamW(baseline_model.parameters(), lr=lr, weight_decay=0.0)
    baseline_scaler = torch.amp.GradScaler()

    stage_3_enabled = config_dict["zero_optimization"]["stage"] == 3
    if stage_3_enabled:
        with deepspeed.zero.Init(config_dict_or_path=config_dict):
            target_model = model_cls(hidden_dim)
        with GatheredParameters(target_model.parameters(), modifier_rank=0):
            for p1, p2 in zip(target_model.parameters(), model.parameters()):
                p1.data.copy_(p2.data)
    else:
        target_model = deepcopy(model)

    ds_optimizer = torch.optim.Adam(target_model.parameters(), lr=lr)
    target_engine, _, _, _ = deepspeed.initialize(config=config_dict, model=target_model, optimizer=ds_optimizer)
    train_batch_size = config_dict["train_micro_batch_size_per_gpu"]

    xs = [torch.randn(train_batch_size, hidden_dim, device=device, dtype=torch.float32) for _ in range(iteration)]
    ys = [torch.randn_like(x) for x in xs]

    for i, (x, y) in enumerate(zip(xs, ys)):
        step_amp(enable, baseline_model, baseline_optimizer, target_engine, dtype, baseline_scaler, x, y, RTOL, ATOL)

    target_engine.destroy()


@pytest.mark.parametrize("enable", [True])
@pytest.mark.parametrize("zero_stage", [3])
@pytest.mark.parametrize("dtype", [torch.bfloat16])
class TestZeroAutoCast(DistributedTest):
    world_size = 2

    def test(self, enable, zero_stage, dtype):
        autocast_conf = {"enabled": enable, "dtype": str(dtype)}

        compare_loss(enable, zero_stage, dtype, autocast_conf)

    def test_safe_modules_conf(self, enable, zero_stage, dtype):
        autocast_conf = {"enabled": enable, "dtype": str(dtype), "lower_precision_safe_modules": ["torch.nn.Linear"]}

        compare_loss(enable, zero_stage, dtype, autocast_conf)
