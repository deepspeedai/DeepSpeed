# Copyright (c) DeepSpeed Team.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
"""BF16 HP-buffer coverage for AutoEP + AutoTP folding correction."""

from types import SimpleNamespace

import torch

from deepspeed.runtime import bf16_optimizer as bf16_mod
from deepspeed.runtime.bf16_optimizer import BF16_Optimizer


def _bf16_optimizer_stub(lp, hp_grad):
    optimizer = object.__new__(BF16_Optimizer)
    optimizer.autoep_folding_spec = SimpleNamespace(tp_size=2, mp_mode="tp")
    optimizer.autoep_folding_tp_group = object()
    optimizer.param_names = {lp: "model.layers.0.mlp.router.gate.weight"}
    optimizer.fp32_groups_gradients = [[hp_grad]]
    optimizer.fp32_groups_has_gradients = [[False]]
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


def test_bf16_immediate_grad_update_hook_reuses_corrected_hp_update(monkeypatch):
    lp = torch.nn.Parameter(torch.ones(2, dtype=torch.bfloat16))
    lp.grad = torch.full((2, ), 6.0, dtype=torch.bfloat16)
    hp_grad = torch.zeros(2, dtype=torch.float32)
    optimizer = _bf16_optimizer_stub(lp, hp_grad)
    optimizer.immediate_grad_update = True

    def fake_apply_folding_correction(_folding_spec, _param, grad, *, tp_group, param_name=None):
        grad.data.div_(3.0)
        _param.ds_autoep_folding_grad_corrected = True
        return "expert_tp_cancel"

    monkeypatch.setattr(bf16_mod,
                        "apply_folding_correction_to_grad_buffer",
                        fake_apply_folding_correction,
                        raising=False)

    optimizer.accumulate_hp_grads_and_remove_lp(lp, group_idx=0, param_idx=0)

    torch.testing.assert_close(hp_grad, torch.full((2, ), 2.0))
    assert lp.ds_autoep_folding_grad_corrected is True
