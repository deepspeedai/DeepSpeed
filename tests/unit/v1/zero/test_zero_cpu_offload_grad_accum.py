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


def _running_grad_norm(engine):
    # The running sum lives in the device partition buffer until a boundary copies it to the host and
    # frees the buffer; after that the host gradient holds it. Groups on the legacy path (Muon, ZenFlow)
    # use the host accumulator.
    optimizer = engine.optimizer
    total_norm_sq = 0.0
    for group_idx in range(len(optimizer.bit16_groups)):
        grad_group = optimizer.averaged_gradients.get(group_idx)
        flat_partition = getattr(grad_group, "flat_partition", None)
        if flat_partition is not None:
            total_norm_sq += flat_partition.float().norm(2).item()**2
        elif group_idx in optimizer._host_owned_compact_partitions:
            host_grad = optimizer.single_partition_of_fp32_groups[group_idx].grad
            total_norm_sq += host_grad.float().norm(2).item()**2
        elif optimizer.accumulated_grads_in_cpu[group_idx] is not None:
            total_norm_sq += optimizer.accumulated_grads_in_cpu[group_idx].float().norm(2).item()**2
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
            norms.append(_running_grad_norm(model))

        model.destroy()

        assert all(norm > 0 for norm in norms), f"running gradient sum should stay non-zero: {norms}"
        for i in range(1, len(norms)):
            assert norms[i] != norms[0], (f"running gradient sum norm did not change after micro-step {i}: "
                                          f"norm[0]={norms[0]:.6f}, norm[{i}]={norms[i]:.6f}. "
                                          "Gradients were not accumulated (PR #7967 regression).")
