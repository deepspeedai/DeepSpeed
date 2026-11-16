# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team
"""Focused correctness tests for ZeRO-3 mixed-dtype gradient accumulation."""

import gc

import pytest
import torch

import deepspeed
from deepspeed.accelerator import get_accelerator
from deepspeed.runtime.zero.stage3 import DeepSpeedZeroOptimizer_Stage3
from unit.common import DistributedTest


def test_mixed_dtype_accumulation_guard_rejects_unsupported_cases():
    eligible = DeepSpeedZeroOptimizer_Stage3._is_mixed_dtype_accumulation_eligible
    assert eligible(True, "cuda", False, False, torch.float16, torch.float32)
    assert not eligible(False, "cuda", False, False, torch.float16, torch.float32)
    assert not eligible(True, "cpu", False, False, torch.float16, torch.float32)
    assert not eligible(True, "cuda", True, False, torch.float16, torch.float32)
    assert not eligible(True, "cuda", False, True, torch.float16, torch.float32)
    assert not eligible(True, "cuda", False, False, torch.bfloat16, torch.float32)
    assert not eligible(True, "cuda", False, False, torch.float32, torch.float32)
    assert not eligible(True, "cuda", False, False, torch.float16, torch.float16)


def _zero3_config(mixed_add):
    return {
        "train_batch_size": 4,
        "train_micro_batch_size_per_gpu": 1,
        "gradient_accumulation_steps": 2,
        "optimizer": {
            "type": "AdamW",
            "params": {
                "lr": 1e-3,
                "torch_adam": True,
            },
        },
        "fp16": {
            "enabled": True,
            "loss_scale": 1,
        },
        "data_types": {
            "grad_accum_dtype": "fp32",
        },
        "zero_optimization": {
            "stage": 3,
            "overlap_comm": True,
            "contiguous_gradients": False,
            "reduce_scatter": True,
            "reduce_bucket_size": 100000,
            "param_persistence_threshold": 0,
            "model_persistence_threshold": 0,
            "experimental_mixed_dtype_grad_accumulation": mixed_add,
        },
    }


@pytest.mark.skipif(get_accelerator().device_name() != "cuda", reason="ZeRO-3 mixed-dtype path requires CUDA/NCCL")
class TestZero3MixedDtypeAccumulation(DistributedTest):
    world_size = 2

    def test_mixed_dtype_two_rank_exact_parity(self, monkeypatch):
        accelerator = get_accelerator()
        device = accelerator.current_device_name()
        eligible_calls = []
        mixed_partition_inputs = []
        original_guard = DeepSpeedZeroOptimizer_Stage3._is_mixed_dtype_accumulation_eligible
        original_partition_grads = DeepSpeedZeroOptimizer_Stage3.partition_grads

        def tracking_guard(enabled, device_name, offload_optimizer, zenflow, source_dtype, destination_dtype):
            is_eligible = original_guard(enabled, device_name, offload_optimizer, zenflow, source_dtype,
                                         destination_dtype)
            if is_eligible:
                eligible_calls.append((source_dtype, destination_dtype))
            return is_eligible

        def record_partition_inputs(optimizer, params, grad_partitions):
            if optimizer.experimental_mixed_dtype_grad_accumulation and optimizer.micro_step_id > 0:
                grad_buffers = optimizer._DeepSpeedZeroOptimizer_Stage3__param_id_to_grad_partition
                for param, grad_partition in zip(params, grad_partitions):
                    destination = grad_buffers[param.ds_id]
                    mixed_partition_inputs.append((optimizer.micro_step_id, grad_partition.dtype, destination.dtype))
            return original_partition_grads(optimizer, params, grad_partitions)

        monkeypatch.setattr(DeepSpeedZeroOptimizer_Stage3, "_is_mixed_dtype_accumulation_eligible",
                            staticmethod(tracking_guard))
        monkeypatch.setattr(DeepSpeedZeroOptimizer_Stage3, "partition_grads", record_partition_inputs)

        def run_variant(mixed_add):
            torch.manual_seed(2718)
            accelerator.manual_seed_all(2718)
            model = torch.nn.Sequential(torch.nn.Linear(5, 7), torch.nn.Tanh(), torch.nn.Linear(7, 3))
            config = _zero3_config(mixed_add)
            engine, _, _, _ = deepspeed.initialize(model=model, model_parameters=model.parameters(), config=config)
            compute_dtype = next(engine.module.parameters()).dtype
            losses = []
            accumulated_gradients = []
            for update in range(2):
                for microstep in range(2):
                    inputs = torch.arange(20, dtype=compute_dtype, device=device).reshape(4, 5)
                    inputs = inputs / 20 + update + microstep
                    loss = engine(inputs).float().square().mean()
                    losses.append(loss.detach().float().cpu())
                    engine.backward(loss)
                accumulated_gradients.append(
                    engine.optimizer.grad_partitions_flat_buffer.detach().float().cpu().clone())
                engine.step()

            parameters = list(engine.module.parameters())
            with deepspeed.zero.GatheredParameters(parameters, modifier_rank=None):
                final_parameters = torch.cat(
                    [parameter.detach().float().reshape(-1).cpu() for parameter in parameters])

            del engine, model
            gc.collect()
            accelerator.synchronize()
            return torch.stack(losses), final_parameters, torch.stack(accumulated_gradients)

        baseline = run_variant(False)
        assert eligible_calls == []
        mixed_add = run_variant(True)
        assert eligible_calls and all(pair == (torch.float16, torch.float32) for pair in eligible_calls)
        assert any(microstep > 0 and source == torch.float16 and destination == torch.float32
                   for microstep, source, destination in mixed_partition_inputs)
        torch.testing.assert_close(mixed_add[0], baseline[0], rtol=0, atol=0)
        torch.testing.assert_close(mixed_add[1], baseline[1], rtol=0, atol=0)
        torch.testing.assert_close(mixed_add[2], baseline[2], rtol=0, atol=0)
