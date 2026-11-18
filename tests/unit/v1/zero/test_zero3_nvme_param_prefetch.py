# Copyright (c) DeepSpeed Team.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
"""ZeRO-3 with every parameter on NVMe (#4072). The parameter trace has an entry per use, so the NVMe prefetch sees
the last layer's parameters for the forward and again for the backward, and with activation checkpointing a
checkpointed layer's parameters for the recompute and again for the backward. It must swap each parameter in once,
and training must match CPU parameter offload."""

import pytest
import torch
from torch.utils.checkpoint import checkpoint

import deepspeed
from deepspeed.accelerator import get_accelerator
from deepspeed.ops.op_builder import AsyncIOBuilder
from unit.common import DistributedTest

# The weights have 2 MiB in bf16, as the NVMe swapper keeps parameters under 1 MiB in CPU memory. Every size is a
# multiple of 512 bytes, so the swaps use direct I/O at aligned addresses.
HIDDEN, BATCH, STEPS = 1024, 4, 4


class _Block(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.fc1 = torch.nn.Linear(HIDDEN, HIDDEN)
        self.fc2 = torch.nn.Linear(HIDDEN, HIDDEN)

    def forward(self, x):
        return x + self.fc2(torch.nn.functional.gelu(self.fc1(x)))


class _Net(torch.nn.Module):

    def __init__(self, checkpointing):
        super().__init__()
        self.checkpointing = checkpointing
        self.inp = torch.nn.Linear(HIDDEN, HIDDEN)
        self.blocks = torch.nn.ModuleList(_Block() for _ in range(3))
        self.out = torch.nn.Linear(HIDDEN, HIDDEN)

    def forward(self, x):
        x = self.inp(x)
        for block in self.blocks:
            if self.checkpointing == "none":
                x = block(x)
            else:
                x = checkpoint(block, x, use_reentrant=self.checkpointing == "reentrant")
        return self.out(x)


def _config(param_device, nvme_path=None):
    offload_param = {"device": param_device}
    if param_device == "nvme":
        # The default stage3_max_reuse_distance keeps all of this small model's weights, and each one holds a swap
        # buffer, so there is a buffer for every weight.
        offload_param.update(nvme_path=nvme_path, max_in_cpu=0, buffer_count=32, buffer_size=HIDDEN * HIDDEN)
    return {
        "train_micro_batch_size_per_gpu": BATCH,
        "steps_per_print": 1000,
        "optimizer": {
            "type": "Adam",
            "params": {
                "lr": 1e-3,
                "torch_adam": True
            }
        },
        "bf16": {
            "enabled": True
        },
        "zero_optimization": {
            "stage": 3,
            "stage3_param_persistence_threshold": 0,
            "offload_param": offload_param,
        },
    }


def _train(config, checkpointing):
    with deepspeed.zero.Init(config_dict_or_path=config):
        torch.manual_seed(0)
        model = _Net(checkpointing)
    engine, _, _, _ = deepspeed.initialize(model=model, model_parameters=model.parameters(), config=config)
    generator = torch.Generator().manual_seed(1)
    losses = []
    for _ in range(STEPS):
        x = torch.randn(BATCH, HIDDEN, generator=generator).to(engine.device, torch.bfloat16)
        y = torch.randint(0, HIDDEN, (BATCH, ), generator=generator).to(engine.device)
        loss = torch.nn.functional.cross_entropy(engine(x).float(), y)
        engine.backward(loss)
        engine.step()
        losses.append(loss.item())
    weights = []
    for param in engine.module.parameters():
        with deepspeed.zero.GatheredParameters([param]):
            weights.append(param.detach().cpu().clone())
    engine.destroy()
    return losses, weights


class TestZero3NVMeParamPrefetch(DistributedTest):
    world_size = 1

    @pytest.mark.parametrize("checkpointing", ["none", "reentrant", "non_reentrant"])
    def test_matches_cpu_offload(self, tmpdir, checkpointing):
        if get_accelerator().device_name() == "cpu":
            pytest.skip("parameter offload needs an accelerator")
        if not deepspeed.ops.__compatible_ops__[AsyncIOBuilder.NAME]:
            pytest.skip("async_io is not compatible")
        if not get_accelerator().is_bf16_supported():
            pytest.skip("bf16 is not supported")

        expected_losses, expected_weights = _train(_config("cpu"), checkpointing)
        losses, weights = _train(_config("nvme", str(tmpdir)), checkpointing)
        assert losses == expected_losses, f"losses {losses}, with CPU offload {expected_losses}"
        for i, (weight, expected) in enumerate(zip(weights, expected_weights)):
            assert torch.equal(weight, expected), f"parameter {i} differs by up to {(weight - expected).abs().max()}"
