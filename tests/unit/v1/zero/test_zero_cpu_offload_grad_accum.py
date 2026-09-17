# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team
"""
Regression test for https://github.com/deepspeedai/DeepSpeed/pull/7967

In ZeRO-2 CPU-offload mode with gradient_accumulation_steps > 1,
`async_accumulate_grad_in_cpu_via_gpu` only copied gradients to CPU
when micro_step_id == 0. For micro_step_id > 0, gradients were
accumulated on GPU but never copied back to CPU, causing
accumulated_grads_in_cpu to stay frozen and the gradient norm to be
underestimated.
"""

import torch
import deepspeed

from unit.common import DistributedTest
from unit.simple_model import SimpleModel, random_dataloader
from deepspeed.accelerator import get_accelerator


def _cpu_grad_norm(engine):
    total_norm_sq = 0.0
    for grad in engine.optimizer.accumulated_grads_in_cpu.values():
        total_norm_sq += grad.float().norm(2).item()**2
    return total_norm_sq**0.5


class TestZero2CPUOffloadGradAccumNorm(DistributedTest):
    world_size = 1

    def test(self):
        gradient_accumulation_steps = 4
        hidden_dim = 10

        config_dict = {
            "train_batch_size": gradient_accumulation_steps,
            "gradient_accumulation_steps": gradient_accumulation_steps,
            "train_micro_batch_size_per_gpu": 1,
            "steps_per_print": 1,
            "zero_optimization": {
                "stage": 2,
                "offload_optimizer": {
                    "device": "cpu",
                },
            },
            "zero_force_ds_cpu_optimizer": False,
            "optimizer": {
                "type": "Adam",
                "params": {
                    "lr": 1e-3,
                },
            },
        }

        if get_accelerator().is_bf16_supported():
            config_dict["bf16"] = {"enabled": True}
        elif get_accelerator().is_fp16_supported():
            config_dict["fp16"] = {"enabled": True}

        torch.manual_seed(42)
        model = SimpleModel(hidden_dim, nlayers=2)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        model, _, _, _ = deepspeed.initialize(
            model=model,
            optimizer=optimizer,
            config=config_dict,
        )

        data_loader = random_dataloader(
            model=model,
            total_samples=gradient_accumulation_steps,
            hidden_dim=hidden_dim,
            device=model.device,
        )

        norms = []
        for batch in data_loader:
            loss = model(batch[0], batch[1])
            model.backward(loss)
            norms.append(_cpu_grad_norm(model))

        model.destroy()

        assert norms[0] > 0, "accumulated_grads_in_cpu should be non-zero after first backward"
        for i in range(1, len(norms)):
            assert norms[i] != norms[0], (f"accumulated_grads_in_cpu norm did not change after micro-step {i}: "
                                          f"norm[0]={norms[0]:.6f}, norm[{i}]={norms[i]:.6f}. "
                                          "Gradients were not copied back to CPU (PR #7967 regression).")
