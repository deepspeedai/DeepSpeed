# Copyright (c) DeepSpeed Team.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
"""BF16 HP-buffer coverage for AutoEP + AutoTP folding correction."""

import contextlib
from types import SimpleNamespace
from unittest import mock

import deepspeed
import deepspeed.comm as dist
import pytest
import torch
from torch.utils.checkpoint import checkpoint

from deepspeed.module_inject.auto_ep_folding import is_autoep_folding_gradient_corrected
from deepspeed.runtime import bf16_optimizer as bf16_mod
from deepspeed.runtime.bf16_optimizer import BF16_Optimizer
from deepspeed.utils import groups, safe_get_full_fp32_param, safe_get_full_grad
from unit.common import DistributedTest
from unit.v1.moe.autoep_test_utils import skip_unless_h100_tests_enabled


def _bf16_optimizer_stub(lp, hp_grad):
    optimizer = object.__new__(BF16_Optimizer)
    optimizer.autoep_folding_spec = SimpleNamespace(tp_size=2, mp_mode="tp")
    optimizer.autoep_folding_tp_group = object()
    optimizer.param_names = {lp: "model.layers.0.mlp.router.gate.weight"}
    optimizer.fp32_groups_gradients = [[hp_grad]]
    optimizer.fp32_groups_has_gradients = [[False]]
    optimizer.graph_harvesting = False
    return optimizer


def test_bf16_hp_grad_update_uses_folded_correction_before_hp_buffer(monkeypatch):
    lp = torch.nn.Parameter(torch.ones(2, dtype=torch.bfloat16))
    lp.grad = torch.full((2, ), 4.0, dtype=torch.bfloat16)
    hp_grad = torch.zeros(2, dtype=torch.float32)
    optimizer = _bf16_optimizer_stub(lp, hp_grad)
    calls = []

    def fake_apply_folding_correction(folding_spec, param, grad, *, tp_group, param_name=None):
        calls.append({
            "folding_spec": folding_spec,
            "param": param,
            "tp_group": tp_group,
            "param_name": param_name,
            "grad_before": grad.detach().float().clone(),
        })
        grad.data.mul_(0.5)
        param.ds_autoep_folding_grad_corrected = True
        return "average"

    monkeypatch.setattr(bf16_mod,
                        "apply_folding_correction_to_grad_buffer",
                        fake_apply_folding_correction,
                        raising=False)

    optimizer._update_hp_grad(lp, group_idx=0, param_idx=0, clear_lp_grads=False)

    assert len(calls) == 1
    assert calls[0]["param"] is lp
    assert calls[0]["param_name"] == "model.layers.0.mlp.router.gate.weight"
    torch.testing.assert_close(calls[0]["grad_before"], torch.full((2, ), 4.0))
    torch.testing.assert_close(hp_grad, torch.full((2, ), 2.0))
    assert optimizer.fp32_groups_has_gradients[0][0] is True
    assert lp.ds_autoep_folding_grad_corrected is True


@pytest.mark.parametrize("micro_batches", [1, 2])
def test_bf16_immediate_grad_update_corrects_every_consumed_gradient(monkeypatch, micro_batches):
    lp = torch.nn.Parameter(torch.ones(2, dtype=torch.bfloat16))
    lp.allreduce = False
    hp_grad = torch.zeros(2, dtype=torch.float32)
    optimizer = _bf16_optimizer_stub(lp, hp_grad)
    optimizer.immediate_grad_update = True
    optimizer.param_names[lp] = "model.layers.0.mlp.experts.w2"
    monkeypatch.setattr(bf16_mod.dist, "get_world_size", lambda group=None: 2)

    for step in range(2):
        expected = 0.0
        for micro_batch in range(micro_batches):
            value = 6.0 + 2 * step + 2 * micro_batch
            lp.grad = torch.full((2, ), value, dtype=torch.bfloat16)
            optimizer.accumulate_hp_grads_and_remove_lp(lp, group_idx=0, param_idx=0)
            expected += value / 2
            torch.testing.assert_close(hp_grad, torch.full_like(hp_grad, expected))
            assert lp.grad is None
            # A consumed gradient must not mark the next micro-batch or step as already corrected.
            assert not is_autoep_folding_gradient_corrected(lp)
        hp_grad.zero_()
        optimizer.fp32_groups_has_gradients[0][0] = False


def test_bf16_immediate_grad_update_preserves_graph_harvesting_addresses(monkeypatch):
    lp = torch.nn.Parameter(torch.ones(2, dtype=torch.bfloat16))
    lp.grad = torch.full((2, ), 6.0, dtype=torch.bfloat16)
    lp.allreduce = False
    hp_grad = torch.zeros(2, dtype=torch.float32)
    optimizer = _bf16_optimizer_stub(lp, hp_grad)
    optimizer.immediate_grad_update = True
    optimizer.graph_harvesting = True
    optimizer.param_names[lp] = "model.layers.0.mlp.experts.w2"
    monkeypatch.setattr(bf16_mod.dist, "get_world_size", lambda group=None: 2)

    optimizer.accumulate_hp_grads_and_remove_lp(lp, group_idx=0, param_idx=0)

    torch.testing.assert_close(hp_grad, torch.full_like(hp_grad, 3.0))
    assert lp.grad is not None
    assert is_autoep_folding_gradient_corrected(lp)


class _FoldedGradientLifecycleModel(torch.nn.Module):

    def __init__(self, expert_scale, reentrant_checkpointing):
        super().__init__()
        self.expert_weight = torch.nn.Parameter(torch.tensor([1.0, -1.0]))
        self.dense = torch.nn.Linear(2, 2, bias=False)
        self.expert_scale = expert_scale
        self.reentrant_checkpointing = reentrant_checkpointing

    def forward(self, x):
        if self.reentrant_checkpointing:
            return checkpoint(self._compute, x, use_reentrant=True)
        return self._compute(x)

    def _compute(self, x):
        return (self.dense(x) + self.expert_weight * self.expert_scale).float().mean()


def _run_bf16_gradient_lifecycle(*, immediate_grad_update, micro_batches):
    torch.manual_seed(7)
    model = _FoldedGradientLifecycleModel(expert_scale=2 if immediate_grad_update else 1,
                                          reentrant_checkpointing=micro_batches == 2)
    config = {
        "train_micro_batch_size_per_gpu": 1,
        "gradient_accumulation_steps": micro_batches,
        "gradient_clipping": 0.25,
        "optimizer": {
            "type": "SGD",
            "params": {
                "lr": 0.1
            }
        },
        "zero_allow_untested_optimizer": True,
        "zero_optimization": {
            "stage": 1
        },
        "bf16": {
            "enabled": True,
            "immediate_grad_update": immediate_grad_update
        },
        "data_types": {
            "grad_accum_dtype": "fp32"
        },
    }
    engine, _, _, _ = deepspeed.initialize(model=model, model_parameters=model.parameters(), config=config)
    assert isinstance(engine.optimizer, BF16_Optimizer)
    expert = engine.module.expert_weight
    tp_group = dist.get_world_group()
    if immediate_grad_update:
        assert dist.get_world_size(group=tp_group) == 2
        # The doubled gradient models folded restore; the plain arm is an independent oracle.
        expert.allreduce = False
        folding_spec = SimpleNamespace(tp_size=2, mp_mode="tp")
        engine._autoep_folding_spec = folding_spec
        engine.optimizer.autoep_folding_spec = folding_spec
        engine.optimizer.autoep_folding_tp_group = tp_group

    tp_group_context = (mock.patch.object(groups, "get_tensor_model_parallel_group", return_value=tp_group)
                        if immediate_grad_update else contextlib.nullcontext())
    steps = []
    previous_master = safe_get_full_fp32_param(expert).detach().float().cpu().clone()
    with tp_group_context:
        for step in range(2):
            gradients = []
            for _ in range(micro_batches):
                x = torch.tensor([[1.0, 2.0]],
                                 dtype=torch.bfloat16,
                                 device=engine.device,
                                 requires_grad=micro_batches == 2)
                loss = engine(x)
                engine.backward(loss)
                accumulated = expert.get_full_hp_grad()
                if immediate_grad_update:
                    assert expert.grad is None
                    public_grad = safe_get_full_grad(expert)
                    assert public_grad is not None and public_grad.dtype == torch.float32
                    torch.testing.assert_close(public_grad, accumulated, rtol=0, atol=0)
                gradients.append(accumulated.detach().float().cpu().clone())
                engine.step()

            norm = float(engine.get_global_grad_norm())
            assert norm > config["gradient_clipping"], f"clipping was not exercised at step {step}: {norm}"
            master = safe_get_full_fp32_param(expert).detach().float().cpu().clone()
            steps.append({"gradients": gradients, "norm": norm, "update": master - previous_master})
            previous_master = master

    engine.destroy()
    return steps


class TestH100BF16FoldingCorrectionLifecycle(DistributedTest):
    world_size = 2
    reuse_dist_env = False

    @pytest.mark.parametrize("micro_batches", [1, 2])
    def test_two_optimizer_steps_match_the_unfolded_reference(self, micro_batches):
        skip_unless_h100_tests_enabled("BF16 TP2 gradient lifecycle needs GPUs")

        folded = _run_bf16_gradient_lifecycle(immediate_grad_update=True, micro_batches=micro_batches)
        reference = _run_bf16_gradient_lifecycle(immediate_grad_update=False, micro_batches=micro_batches)
        for step, (actual, expected) in enumerate(zip(folded, reference)):
            for micro_batch, (actual_grad, expected_grad) in enumerate(zip(actual["gradients"],
                                                                           expected["gradients"])):
                assert expected_grad.norm() > 0
                torch.testing.assert_close(actual_grad,
                                           expected_grad,
                                           rtol=1e-3,
                                           atol=1e-6,
                                           msg=f"expert gradient at step {step}, micro-batch {micro_batch}")
            assert expected["update"].norm() > 0
            torch.testing.assert_close(actual["update"],
                                       expected["update"],
                                       rtol=1e-3,
                                       atol=1e-6,
                                       msg=f"clipped expert update at step {step}")
            assert abs(actual["norm"] - expected["norm"]) / expected["norm"] <= 1e-3
