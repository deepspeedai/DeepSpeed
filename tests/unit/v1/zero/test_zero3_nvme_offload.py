# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
"""ZeRO-3 with NVMe offload must train exactly like CPU offload: DeepSpeedCPUAdam steps on the same values, NVMe only
adds the swapping. The model's sizes make direct I/O on the partitions in the flat buffers hard: the first weight's
length is not a whole number of I/O blocks, and the odd-sized parameters before the second weight move its partition
to an address that direct I/O cannot use."""

import pytest
import torch

import deepspeed
from deepspeed.accelerator import get_accelerator
from deepspeed.ops.op_builder import AsyncIOBuilder, CPUAdamBuilder
from unit.common import DistributedTest

IN, ODD, HIDDEN = 1000, 999, 1024
BATCH, STEPS = 4, 3


def _skip_if_unsupported():
    if not deepspeed.ops.__compatible_ops__[AsyncIOBuilder.NAME]:
        pytest.skip("async_io is not compatible")
    if not deepspeed.ops.__compatible_ops__[CPUAdamBuilder.NAME]:
        pytest.skip("cpu_adam is not compatible")
    if not get_accelerator().is_bf16_supported():
        pytest.skip("bf16 is not supported")


def _config(nvme_path, optimizer_device, param_device, pipeline=False, fast_init=False):
    offload_optimizer = {"device": optimizer_device}
    if optimizer_device == "nvme":
        offload_optimizer.update(nvme_path=nvme_path,
                                 pipeline_read=pipeline,
                                 pipeline_write=pipeline,
                                 buffer_count=12 if pipeline else 4,
                                 fast_init=fast_init)
    offload_param = {"device": param_device}
    if param_device == "nvme":
        offload_param.update(nvme_path=nvme_path, buffer_count=4, buffer_size=int(4e6))
        if fast_init:
            # Parameters only on NVMe, so the fp32 copies are made from the swap files.
            offload_param.update(max_in_cpu=0)
    return {
        "train_micro_batch_size_per_gpu": BATCH,
        "steps_per_print": 1000,
        "optimizer": {
            "type": "Adam",
            "params": {
                "lr": 1e-3
            }
        },
        "bf16": {
            "enabled": True
        },
        "zero_optimization": {
            "stage": 3,
            "stage3_param_persistence_threshold": 0,
            "stage3_prefetch_bucket_size": 0,
            "offload_optimizer": offload_optimizer,
            "offload_param": offload_param,
        },
    }


def _train(config):
    with deepspeed.zero.Init(config_dict_or_path=config):
        torch.manual_seed(0)
        model = torch.nn.Sequential(torch.nn.Linear(IN, ODD), torch.nn.LayerNorm(ODD), torch.nn.Linear(ODD, HIDDEN),
                                    torch.nn.GELU(), torch.nn.Linear(HIDDEN, 10))
    engine, _, _, _ = deepspeed.initialize(model=model, model_parameters=model.parameters(), config=config)
    generator = torch.Generator().manual_seed(1)
    losses = []
    for _ in range(STEPS):
        x = torch.randn(BATCH, IN, generator=generator).to(engine.device, torch.bfloat16)
        y = torch.randint(0, 10, (BATCH, ), generator=generator).to(engine.device)
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


def _assert_trains_like_cpu_offload(tmpdir, optimizer_device, param_device, **nvme_options):
    expected_losses, expected_weights = _train(_config(None, "cpu", "cpu" if param_device == "nvme" else "none"))
    losses, weights = _train(_config(str(tmpdir), optimizer_device, param_device, **nvme_options))
    assert losses == expected_losses, f"losses {losses}, with CPU offload {expected_losses}"
    for i, (weight, expected) in enumerate(zip(weights, expected_weights)):
        assert torch.equal(weight, expected), f"parameter {i} differs by up to {(weight - expected).abs().max()}"


class TestZero3NVMeOffload(DistributedTest):
    world_size = 1

    @pytest.mark.parametrize("pipeline", [False, True])
    def test_optimizer(self, tmpdir, pipeline):
        _skip_if_unsupported()
        _assert_trains_like_cpu_offload(tmpdir, "nvme", "none", pipeline=pipeline)

    def test_param(self, tmpdir):
        _skip_if_unsupported()
        _assert_trains_like_cpu_offload(tmpdir, "cpu", "nvme")

    @pytest.mark.parametrize("fast_init", [False, True])
    def test_param_and_optimizer(self, tmpdir, fast_init):
        _skip_if_unsupported()
        _assert_trains_like_cpu_offload(tmpdir, "nvme", "nvme", fast_init=fast_init)
