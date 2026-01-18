# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# Debug script to investigate testRowParallel numerical differences

import pytest
import deepspeed.comm as dist
import torch
from copy import deepcopy

from unit.common import DistributedTest, preferred_dtype
import deepspeed
from deepspeed.accelerator import get_accelerator
from deepspeed.utils import groups
from torch import nn
from deepspeed.module_inject.layers import LinearAllreduce, set_autotp_mode


def process_linear_layer(hidden_dim, input, seed=42):
    torch.manual_seed(seed)
    torch_linear = nn.Linear(hidden_dim,
                             hidden_dim,
                             dtype=preferred_dtype(),
                             device=get_accelerator().current_device())
    torch_out = torch_linear(input)
    torch_loss = torch_out.sum()
    torch_loss.backward()
    return torch_linear, torch_out


@pytest.mark.parametrize("seed", [42, 123, 456, 789, 1024])
class TestRowParallelDebug(DistributedTest):
    world_size = 2
    reuse_dist_env = False

    def test(self, seed: int):
        tp_size = 2
        tp_overlap_comm = False
        hidden_dim = 128
        batch_size_per_device = 1
        set_autotp_mode(training=True)
        config_dict = {
            "train_micro_batch_size_per_gpu": 1,
            "steps_per_print": 1,
            "optimizer": {
                "type": "Adam",
                "params": {
                    "lr": 1e-6
                }
            },
            "tensor_parallel": {
                "autotp_size": tp_size,
                "tp_overlap_comm": tp_overlap_comm
            },
            "zero_optimization": {
                "stage": 0,
            }
        }
        if preferred_dtype() is torch.float16:
            config_dict["fp16"] = {"enabled": True}
        elif preferred_dtype() is torch.bfloat16:
            config_dict["bf16"] = {"enabled": True}

        from unit.simple_model import SimpleModel
        model = SimpleModel(hidden_dim=hidden_dim)
        model, _, _, _ = deepspeed.initialize(model=model, model_parameters=model.parameters(), config=config_dict)
        
        # Use the seed for input generation
        torch.manual_seed(seed)
        input = torch.randn(batch_size_per_device,
                            hidden_dim,
                            dtype=preferred_dtype(),
                            requires_grad=True,
                            device=get_accelerator().current_device())

        dist.broadcast(input,
                       groups.get_tensor_model_parallel_src_rank(),
                       group=groups.get_tensor_model_parallel_group())
        
        # Barrier to ensure broadcast completes
        dist.barrier()

        torch_linear, torch_out = process_linear_layer(hidden_dim, input, seed=seed)
        linear = LinearAllreduce(deepcopy(torch_linear), groups.get_tensor_model_parallel_group())

        input_ = torch.chunk(input, tp_size, dim=-1)[groups.get_tensor_model_parallel_rank()]
        out = linear(input_.to(get_accelerator().current_device()))
        loss = out.sum()
        loss.backward()

        torch_grad = torch.chunk(torch_linear.weight.grad, tp_size, dim=1)[groups.get_tensor_model_parallel_rank()]
        torch_bias_grad = torch_linear.bias.grad
        
        # Calculate differences
        bias_grad_diff = (linear.bias.grad - torch_bias_grad.to(get_accelerator().current_device())).abs()
        weight_grad_diff = (linear.weight.grad - torch_grad.to(get_accelerator().current_device())).abs()
        out_diff = (out - torch_out.to(get_accelerator().current_device())).abs()
        
        rank = dist.get_rank()
        print(f"\n=== Seed {seed}, Rank {rank} ===")
        print(f"Preferred dtype: {preferred_dtype()}")
        print(f"Bias grad - max diff: {bias_grad_diff.max().item():.6e}, mean diff: {bias_grad_diff.mean().item():.6e}")
        print(f"Weight grad - max diff: {weight_grad_diff.max().item():.6e}, mean diff: {weight_grad_diff.mean().item():.6e}")
        print(f"Output - max diff: {out_diff.max().item():.6e}, mean diff: {out_diff.mean().item():.6e}")
        print(f"Output shape: {out.shape}, torch_out shape: {torch_out.shape}")
        print(f"Output sample values: {out[0, :5].tolist()}")
        print(f"Torch out sample values: {torch_out[0, :5].tolist()}")
        
        # Check assertions with detailed error messages
        bias_close = torch.allclose(linear.bias.grad, torch_bias_grad.to(get_accelerator().current_device()), atol=1e-3)
        weight_close = torch.allclose(linear.weight.grad, torch_grad.to(get_accelerator().current_device()), atol=1e-3)
        out_close = torch.allclose(out, torch_out.to(get_accelerator().current_device()), atol=1e-2)
        
        print(f"Bias grad close (atol=1e-3): {bias_close}")
        print(f"Weight grad close (atol=1e-3): {weight_close}")
        print(f"Output close (atol=1e-2): {out_close}")
        
        assert bias_close, f"Bias grad max diff: {bias_grad_diff.max().item()}"
        assert weight_close, f"Weight grad max diff: {weight_grad_diff.max().item()}"
        assert out_close, f"Output max diff: {out_diff.max().item()}"
