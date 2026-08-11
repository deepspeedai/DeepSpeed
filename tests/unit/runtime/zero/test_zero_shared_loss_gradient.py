# Copyright (c) DeepSpeed Team.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import pytest
import torch

import deepspeed
import deepspeed.comm as dist
from deepspeed.accelerator import get_accelerator
from deepspeed.utils import safe_get_full_grad
from unit.common import DistributedTest
from unit.util import bf16_required_version_check


class SharedLinear(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.shared = torch.nn.Linear(4, 4, bias=False)

    def forward(self, inputs):
        return self.shared(inputs)


def _make_model(device):
    model = SharedLinear().to(device=device, dtype=torch.bfloat16)
    with torch.no_grad():
        values = torch.arange(16, device=device, dtype=torch.float32).reshape(4, 4) / 16
        model.shared.weight.copy_(values.to(torch.bfloat16))
    return model


def _inputs(device, rank, micro_step):
    values = torch.arange(8, device=device, dtype=torch.float32).reshape(2, 4)
    return (values + rank * 3 + micro_step).to(torch.bfloat16)


class TestZero2SharedLossGradient(DistributedTest):
    world_size = 2

    def test_engine_and_module_branches_match_manual_reference(self):
        if not bf16_required_version_check():
            pytest.skip("BF16 ZeRO-2 test requires BF16 accelerator support.")

        gradient_accumulation_steps = 8
        device = get_accelerator().current_device_name()
        rank = dist.get_rank()
        world_size = dist.get_world_size()

        reference_model = _make_model(device)
        reference_grad = torch.zeros_like(reference_model.shared.weight, dtype=torch.float32)
        for micro_step in range(gradient_accumulation_steps):
            reference_model.zero_grad(set_to_none=True)
            inputs = _inputs(device, rank, micro_step)
            secondary_inputs = inputs * 0.5
            output = reference_model(inputs) + reference_model(secondary_inputs)
            loss = output.float().square().mean() / gradient_accumulation_steps
            loss.backward()

            microbatch_grad = reference_model.shared.weight.grad.detach().clone()
            dist.all_reduce(microbatch_grad)
            microbatch_grad.div_(world_size)
            reference_grad.add_(microbatch_grad.float())

        model = _make_model(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        config = {
            "train_micro_batch_size_per_gpu": 2,
            "gradient_accumulation_steps": gradient_accumulation_steps,
            "bf16": {
                "enabled": True,
            },
            "zero_allow_untested_optimizer": True,
            "zero_optimization": {
                "stage": 2,
                "overlap_comm": True,
                "contiguous_gradients": True,
                "reduce_scatter": True,
            },
        }
        engine, *_ = deepspeed.initialize(model=model, optimizer=optimizer, config=config)
        try:
            for micro_step in range(gradient_accumulation_steps):
                inputs = _inputs(device, rank, micro_step)
                secondary_inputs = inputs * 0.5
                output = engine(inputs) + engine.module(secondary_inputs)
                engine.backward(output.float().square().mean())
                if micro_step + 1 < gradient_accumulation_steps:
                    engine.step()

            actual_grad = safe_get_full_grad(engine.module.shared.weight)
            assert actual_grad is not None
            torch.testing.assert_close(actual_grad.float(), reference_grad, rtol=5e-3, atol=2.0)
        finally:
            engine.destroy()
