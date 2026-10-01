# Copyright (c) DeepSpeed Team.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import pytest
import torch
from torch.utils.checkpoint import checkpoint

import deepspeed
import deepspeed.comm as dist
from deepspeed.accelerator import get_accelerator
from deepspeed.runtime.engine import _ENGINE_BACKWARD_GRAPH_TRACKER
from deepspeed.utils import safe_get_full_grad
from unit.common import DistributedTest
from unit.util import bf16_required_version_check


class SharedLinear(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.shared = torch.nn.Linear(4, 4, bias=False)

    def forward(self, inputs):
        return self.shared(inputs)


class _RaiseInBackward(torch.autograd.Function):

    @staticmethod
    def forward(ctx, inputs):
        return inputs.clone()

    @staticmethod
    def backward(ctx, grad):
        raise RuntimeError("intentional backward failure")


class FailingSharedLinear(SharedLinear):

    def forward(self, inputs):
        return _RaiseInBackward.apply(super().forward(inputs))


def _make_model(device, model_class=SharedLinear):
    model = model_class().to(device=device, dtype=torch.bfloat16)
    with torch.no_grad():
        values = torch.arange(16, device=device, dtype=torch.float32).reshape(4, 4) / 16
        model.shared.weight.copy_(values.to(torch.bfloat16))
    return model


def _inputs(device, rank, micro_step):
    values = torch.arange(8, device=device, dtype=torch.float32).reshape(2, 4)
    return (values + rank * 3 + micro_step).to(torch.bfloat16)


def _make_engine(model, gradient_accumulation_steps):
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
    return engine


def _reference_grad(model, inputs):
    model.zero_grad(set_to_none=True)
    model(inputs).float().square().mean().backward()
    grad = model.shared.weight.grad.detach().clone()
    dist.all_reduce(grad)
    grad.div_(dist.get_world_size())
    return grad.float()


def _advance_to_accumulation_boundary(engine, inputs, gradient_accumulation_steps):
    for _ in range(gradient_accumulation_steps - 1):
        engine.backward(engine(inputs).float().sum() * 0)
        engine.step()


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

        engine = _make_engine(_make_model(device), gradient_accumulation_steps)
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

    @pytest.mark.parametrize("graph_task_ids_available", [True, False], ids=["graph-task-id", "fallback"])
    def test_two_engine_managed_backward_scales_each_branch_once(self, monkeypatch, graph_task_ids_available):
        if not bf16_required_version_check():
            pytest.skip("BF16 ZeRO-2 test requires BF16 accelerator support.")

        gradient_accumulation_steps = 8
        device = get_accelerator().current_device_name()
        rank = dist.get_rank()
        inputs = [_inputs(device, rank, micro_step) for micro_step in range(2)]
        reference_grads = [_reference_grad(_make_model(device), value) for value in inputs]
        engines = [_make_engine(_make_model(device), gradient_accumulation_steps) for _ in range(2)]
        try:
            monkeypatch.setattr(type(_ENGINE_BACKWARD_GRAPH_TRACKER), "supported",
                                staticmethod(lambda: graph_task_ids_available))
            for engine, value in zip(engines, inputs):
                _advance_to_accumulation_boundary(engine, value, gradient_accumulation_steps)
            losses = [engine(value).float().square().mean() for engine, value in zip(engines, inputs)]
            engines[0].backward(sum(losses))

            for index, (engine, reference_grad) in enumerate(zip(engines, reference_grads)):
                actual_grad = safe_get_full_grad(engine.module.shared.weight)
                assert actual_grad is not None
                torch.testing.assert_close(actual_grad.float(),
                                           reference_grad / gradient_accumulation_steps,
                                           rtol=5e-3,
                                           atol=2.0,
                                           msg=f"engine {index} gradient was not scaled exactly once")
        finally:
            for engine in engines:
                engine.destroy()

    def test_reentrant_checkpointed_engine_branches_are_scaled_once(self):
        if not bf16_required_version_check():
            pytest.skip("BF16 ZeRO-2 test requires BF16 accelerator support.")

        gradient_accumulation_steps = 8
        device = get_accelerator().current_device_name()
        rank = dist.get_rank()
        inputs = [_inputs(device, rank, micro_step) for micro_step in range(2)]
        reference_grads = [_reference_grad(_make_model(device), value) for value in inputs]
        engines = [_make_engine(_make_model(device), gradient_accumulation_steps) for _ in range(2)]
        try:
            for engine, value in zip(engines, inputs):
                _advance_to_accumulation_boundary(engine, value, gradient_accumulation_steps)
            losses = [
                checkpoint(engine, value.detach().requires_grad_(True), use_reentrant=True).float().square().mean()
                for engine, value in zip(engines, inputs)
            ]
            engines[0].backward(sum(losses))

            for index, (engine, reference_grad) in enumerate(zip(engines, reference_grads)):
                actual_grad = safe_get_full_grad(engine.module.shared.weight)
                assert actual_grad is not None
                torch.testing.assert_close(actual_grad.float(),
                                           reference_grad / gradient_accumulation_steps,
                                           rtol=5e-3,
                                           atol=2.0,
                                           msg=f"reentrant engine {index} gradient was not scaled exactly once")
        finally:
            for engine in engines:
                engine.destroy()

    def test_managed_backward_context_restored_after_exception(self):
        if not bf16_required_version_check():
            pytest.skip("BF16 ZeRO-2 test requires BF16 accelerator support.")

        gradient_accumulation_steps = 8
        device = get_accelerator().current_device_name()
        rank = dist.get_rank()
        inputs = _inputs(device, rank, 0)
        reference_grad = _reference_grad(_make_model(device), inputs)
        failing_engine = _make_engine(_make_model(device, FailingSharedLinear), gradient_accumulation_steps)
        direct_backward_engine = _make_engine(_make_model(device), gradient_accumulation_steps)
        try:
            _advance_to_accumulation_boundary(direct_backward_engine, inputs, gradient_accumulation_steps)
            failing_loss = failing_engine(inputs).float().square().mean()
            with pytest.raises(RuntimeError, match="intentional backward failure"):
                failing_engine.backward(failing_loss)
            assert failing_engine._running_engine_backward is False
            assert _ENGINE_BACKWARD_GRAPH_TRACKER.active_registration_count() == 0

            direct_backward_engine(inputs).float().square().mean().backward()
            actual_grad = safe_get_full_grad(direct_backward_engine.module.shared.weight)
            assert actual_grad is not None
            torch.testing.assert_close(actual_grad.float(),
                                       reference_grad / gradient_accumulation_steps,
                                       rtol=5e-3,
                                       atol=2.0)
        finally:
            failing_engine.destroy()
            direct_backward_engine.destroy()
