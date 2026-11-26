# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team
"""A missing ZeRO fragment must fail on every rank.

safe_get_full_optimizer_state() and safe_get_full_grad() gather with a
collective. Raising only on ranks that own a fragment lets the others enter
that collective, so the job splits. These tests check that every rank raises,
and that a complete fragment still gathers to the same value.
"""

import pytest
import torch

import deepspeed
import deepspeed.comm as dist
from deepspeed.accelerator import get_accelerator
from deepspeed.utils import safe_get_full_grad, safe_get_full_optimizer_state
from unit.common import DistributedTest


class _WeightAndLora(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(4, 4))
        # One element stays inside a single ZeRO-1/2 partition after padding.
        self.lora = torch.nn.Parameter(torch.zeros(1))

    def forward(self, x):
        return (self.weight * x).sum() + self.lora.sum()


def _fragment_config(stage):
    config = {
        "train_micro_batch_size_per_gpu": 1,
        "steps_per_print": 1,
        "optimizer": {
            "type": "Adam",
            "params": {
                "lr": 1e-3,
                "torch_adam": True,
            },
        },
        "bf16": {
            "enabled": True
        },
        "zero_optimization": {
            "stage": stage,
        },
    }
    if stage == 3:
        config["zero_optimization"]["param_persistence_threshold"] = 0
    return config


def _owner_count(param):
    # NCCL rejects a CPU tensor. current_device_name() is 'cpu' on the CPU accelerator.
    device = get_accelerator().current_device_name()
    owned = torch.zeros(1, dtype=torch.float32, device=device)
    if getattr(param, "_hp_mapping", None) is not None:
        owned[0] = 1
    dist.all_reduce(owned)
    return int(owned.item())


def _assert_same_across_ranks(value):
    reference = value.detach().clone()
    dist.broadcast(reference, src=0)
    torch.testing.assert_close(value, reference)


class TestMissingFragmentAgrees(DistributedTest):
    world_size = 2

    def test_zero2_missing_state_and_grad_raise_on_every_rank(self):
        if not get_accelerator().is_bf16_supported():
            pytest.skip("requires bf16")

        model = _WeightAndLora()
        # One group so the step links state for every parameter. The 1-element
        # parameter still sits on a single partition.
        param_groups = [{"params": [model.weight, model.lora]}]
        engine, _, _, _ = deepspeed.initialize(model=model, model_parameters=param_groups, config=_fragment_config(2))
        try:
            weight = engine.module.weight
            lora = engine.module.lora
            assert _owner_count(lora) == 1
            assert _owner_count(weight) == dist.get_world_size()

            with pytest.raises(ValueError, match="exp_avg"):
                safe_get_full_optimizer_state(lora, "exp_avg")
            with pytest.raises(ValueError, match="exp_avg"):
                safe_get_full_optimizer_state(weight, "exp_avg")
            with pytest.raises(ValueError, match="Gradients are only available"):
                safe_get_full_grad(lora)

            x = torch.ones(4, 4, dtype=weight.dtype, device=engine.device)
            engine.backward(engine(x))
            _assert_same_across_ranks(safe_get_full_grad(lora))
            _assert_same_across_ranks(safe_get_full_grad(weight))

            engine.step()
            lora_state = safe_get_full_optimizer_state(lora, "exp_avg")
            weight_state = safe_get_full_optimizer_state(weight, "exp_avg")
            assert lora_state.shape == lora.shape
            assert weight_state.shape == weight.shape
            _assert_same_across_ranks(lora_state)
            _assert_same_across_ranks(weight_state)

            with pytest.raises(ValueError, match="Gradients are only available"):
                safe_get_full_grad(lora)

            if dist.get_rank() == 0:
                weight._hp_mapping.optim_fragment.pop("exp_avg")
            with pytest.raises(ValueError, match="exp_avg"):
                safe_get_full_optimizer_state(weight, "exp_avg")
        finally:
            engine.destroy()

    def test_zero3_missing_state_raises_on_every_rank(self):
        if not get_accelerator().is_bf16_supported():
            pytest.skip("requires bf16")

        model = _WeightAndLora()
        param_groups = [{"params": [model.weight]}, {"params": [model.lora]}]
        engine, _, _, _ = deepspeed.initialize(model=model, model_parameters=param_groups, config=_fragment_config(3))
        try:
            weight = engine.module.weight
            with pytest.raises(ValueError, match="exp_avg"):
                safe_get_full_optimizer_state(weight, "exp_avg")

            x = torch.ones(4, 4, dtype=weight.dtype, device=engine.device)
            engine.backward(engine(x))
            engine.step()
            state = safe_get_full_optimizer_state(weight, "exp_avg")
            assert state is not None
            assert tuple(state.shape) == tuple(getattr(weight, "ds_shape", weight.shape))
            _assert_same_across_ranks(state)

            zero = engine.optimizer
            group_idx = zero.grad_position[zero.get_param_id(weight)][0]
            fp32_param = zero.fp32_partitioned_groups_flat[group_idx]
            if dist.get_rank() == 0:
                del zero.optimizer.state[fp32_param]["exp_avg"]
            with pytest.raises(ValueError, match="exp_avg"):
                safe_get_full_optimizer_state(weight, "exp_avg")
        finally:
            engine.destroy()
