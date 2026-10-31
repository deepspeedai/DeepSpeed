# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import os

import pytest
import torch

import deepspeed
from deepspeed.accelerator import get_accelerator
from deepspeed.runtime.bf16_optimizer import BF16_Optimizer
from deepspeed.runtime.zero.stage3 import DeepSpeedZeroOptimizer_Stage3
from unit.common import DistributedTest

TAG = "fp32-masters"


class _OneParam(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(4, 4))

    def forward(self, x):
        return (x @ self.weight).sum()


def _config(kind):
    config = {
        "train_micro_batch_size_per_gpu": 1,
        "train_batch_size": 1,
        "gradient_accumulation_steps": 1,
        "optimizer": {
            "type": "Adam",
            "params": {
                "lr": 0.1
            },
        },
        "bf16": {
            "enabled": True
        },
        "zero_optimization": {
            "stage": 3 if kind == "zero3" else 1
        },
    }
    if kind == "bf16":
        # Stage 1 with bf16 parameters and fp32 gradient accumulation selects BF16_Optimizer.
        config["data_types"] = {"grad_accum_dtype": "fp32"}
    else:
        config["zero_optimization"]["stage3_param_persistence_threshold"] = 0
    return config


def _init_engine(kind):
    model = _OneParam()
    engine, _, _, _ = deepspeed.initialize(config=_config(kind), model=model, model_parameters=model.parameters())
    if kind == "bf16":
        assert isinstance(engine.optimizer, BF16_Optimizer)
    else:
        assert isinstance(engine.optimizer, DeepSpeedZeroOptimizer_Stage3)
    return engine


def _masters(engine):
    optimizer = engine.optimizer
    flats = (optimizer.fp32_groups_flat_partition
             if isinstance(optimizer, BF16_Optimizer) else optimizer.fp32_partitioned_groups_flat)
    assert len(flats) == 1
    return flats[0].detach()


def _has_moments(engine):
    return any("exp_avg" in state for state in engine.optimizer.optimizer.state.values())


def _train_and_detach_masters(engine):
    data = torch.ones(1, 4, device=engine.device, dtype=torch.bfloat16)
    engine.backward(engine(data))
    engine.step()
    with torch.no_grad():
        _masters(engine).add_(2**-12)
    assert _has_moments(engine)
    saved = _masters(engine).detach().clone()
    assert not torch.equal(saved, saved.to(torch.bfloat16).to(saved.dtype))
    return saved


class TestFp32MastersStage3AndBf16(DistributedTest):
    world_size = 1
    backend = "gloo"
    requires_cuda_env = False

    @pytest.mark.parametrize("kind", ["bf16", "zero3"])
    def test_masters_only_keeps_saved_precision(self, tmpdir, kind):
        if not get_accelerator().is_bf16_supported():
            pytest.skip("bf16 is not supported")

        engine = _init_engine(kind)
        saved = _train_and_detach_masters(engine)
        load_dir = os.path.join(str(tmpdir), kind)
        engine.save_checkpoint(load_dir, tag=TAG)

        loaded = _init_engine(kind)
        loaded.load_checkpoint(load_dir,
                               tag=TAG,
                               load_module_only=False,
                               load_optimizer_states=False,
                               load_lr_scheduler_states=False)
        assert torch.equal(_masters(loaded).cpu(), saved.cpu())
        assert not _has_moments(loaded)

        module_only = _init_engine(kind)
        module_only.load_checkpoint(load_dir, tag=TAG, load_module_only=True)
        rebuilt = _masters(module_only).cpu()
        # ZeRO-3 stores the partitioned weights in the fp32 shard, so module-only still reads them.
        # The dedicated BF16 optimizer rebuilds its masters from the loaded bf16 weights.
        if kind == "zero3":
            assert torch.equal(rebuilt, saved.cpu())
        else:
            assert torch.equal(rebuilt, saved.to(torch.bfloat16).to(saved.dtype).cpu())
        assert not _has_moments(module_only)
