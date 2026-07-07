# Copyright (c) DeepSpeed Team.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
"""ZeRO-1 overlap hook coverage for AutoEP + AutoTP folding correction."""

from types import SimpleNamespace

import torch

from deepspeed.runtime.zero import stage_1_and_2 as zero_mod
from deepspeed.runtime.zero.stage_1_and_2 import DeepSpeedZeroOptimizer


def _zero_optimizer_stub(*, partition_gradients, overlap_comm):
    optimizer = object.__new__(DeepSpeedZeroOptimizer)
    optimizer.partition_gradients = partition_gradients
    optimizer.overlap_comm = overlap_comm
    optimizer.autoep_folding_tp_group = object()
    optimizer.autoep_folding_spec = SimpleNamespace(tp_size=2, mp_mode="tp")
    return optimizer


def test_zero1_overlap_applies_folding_before_hook_consumes_grad(monkeypatch):
    optimizer = _zero_optimizer_stub(partition_gradients=False, overlap_comm=True)
    param = torch.nn.Parameter(torch.ones(2))
    grad = torch.full((2, ), 4.0)
    calls = []

    def fake_apply_folding_correction(folding_spec,
                                      param,
                                      grad,
                                      *,
                                      tp_group,
                                      param_name=None,
                                      use_correction_marker=True):
        calls.append({
            "folding_spec": folding_spec,
            "param": param,
            "tp_group": tp_group,
            "grad_before": grad.detach().clone(),
            "use_correction_marker": use_correction_marker,
        })
        grad.data.mul_(0.5)
        param.ds_autoep_folding_grad_corrected = True
        return "average"

    monkeypatch.setattr(zero_mod,
                        "apply_folding_correction_to_grad_buffer",
                        fake_apply_folding_correction,
                        raising=False)

    optimizer._maybe_reduce_autoep_folding_tp_gradient(param, grad)

    assert len(calls) == 1
    assert calls[0]["param"] is param
    assert calls[0]["use_correction_marker"] is True
    torch.testing.assert_close(calls[0]["grad_before"], torch.full((2, ), 4.0))
    torch.testing.assert_close(grad, torch.full((2, ), 2.0))
    assert param.ds_autoep_folding_grad_corrected is True


def test_zero1_nonoverlap_leaves_engine_boundary_sweep_owner(monkeypatch):
    optimizer = _zero_optimizer_stub(partition_gradients=False, overlap_comm=False)
    param = torch.nn.Parameter(torch.ones(2))
    grad = torch.full((2, ), 4.0)
    calls = []
    monkeypatch.setattr(zero_mod,
                        "apply_folding_correction_to_grad_buffer",
                        lambda *args, **kwargs: calls.append((args, kwargs)),
                        raising=False)

    optimizer._maybe_reduce_autoep_folding_tp_gradient(param, grad)

    assert calls == []
    torch.testing.assert_close(grad, torch.full((2, ), 4.0))


def test_zero2_partitioned_path_still_applies_folding(monkeypatch):
    optimizer = _zero_optimizer_stub(partition_gradients=True, overlap_comm=False)
    param = torch.nn.Parameter(torch.ones(2))
    grad = torch.full((2, ), 4.0)
    calls = []

    def fake_apply_folding_correction(_folding_spec,
                                      _param,
                                      grad,
                                      *,
                                      tp_group,
                                      param_name=None,
                                      use_correction_marker=True):
        calls.append((_param, use_correction_marker))
        grad.data.div_(2.0)
        return "expert_tp_cancel"

    monkeypatch.setattr(zero_mod,
                        "apply_folding_correction_to_grad_buffer",
                        fake_apply_folding_correction,
                        raising=False)

    optimizer._maybe_reduce_autoep_folding_tp_gradient(param, grad)

    assert calls == [(param, False)]
    torch.testing.assert_close(grad, torch.full((2, ), 2.0))
