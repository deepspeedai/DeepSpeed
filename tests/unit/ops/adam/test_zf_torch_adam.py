# Copyright (c) DeepSpeed Team.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import pytest
import torch
import numpy as np
from torch.nn import Parameter
from deepspeed.ops.adam import ZenFlowSelectiveAdamW, ZenFlowSelectiveAdamW_stage3


def make_param(Opt, shape, selected_indices=None):
    param = Parameter(torch.randn(*shape))

    if Opt is ZenFlowSelectiveAdamW_stage3:
        if param.dim() == 2:
            param.ds_shape = (param.shape[1], param.shape[0])
            param.ds_tensor = param.clone().T.contiguous().view(-1)
        else:
            param.ds_shape = tuple(param.shape)
            param.ds_tensor = param.clone()

        param.complete_column_offset = 0
        param.complete_numel = param.numel()
        param.group_id = 0

    if selected_indices is not None:
        param.selected_indices = selected_indices
        if param.dim() == 2:
            param.selected_grad = torch.randn(
                param.shape[0], len(selected_indices)) if Opt is not ZenFlowSelectiveAdamW_stage3 else torch.randn(
                    len(selected_indices), param.ds_shape[1])
            param.temp_selected_param = param.data[:, selected_indices].clone(
            ) if Opt is not ZenFlowSelectiveAdamW_stage3 else param.ds_tensor.view(
                param.ds_shape)[selected_indices, :].clone()
        else:
            param.selected_grad = torch.randn_like(param.data)
            param.temp_selected_param = param.data.clone()
    return param


@pytest.mark.parametrize("Opt", [ZenFlowSelectiveAdamW, ZenFlowSelectiveAdamW_stage3])
def test_init_methods(Opt):
    opt1 = Opt([torch.nn.Parameter(torch.randn(2, 4))], lr=1e-3, offload=False)
    assert opt1.step == opt1._step_without_offload
    opt2 = Opt([torch.nn.Parameter(torch.randn(2, 4))], lr=1e-3, offload=True)
    assert opt2.step == opt2._step_with_offload


@pytest.mark.parametrize("Opt", [ZenFlowSelectiveAdamW, ZenFlowSelectiveAdamW_stage3])
def test_step_without_offload(Opt):
    param = make_param(Opt, (4, 6), torch.tensor([1, 3, 4]))
    param.requires_grad_(True)
    opt = Opt([param], lr=1e-3, offload=False)

    old_selected = param.data[:, param.selected_indices].clone(
    ) if Opt is not ZenFlowSelectiveAdamW_stage3 else param.ds_tensor.view(
        param.ds_shape)[param.selected_indices, :].clone()
    opt.step()
    new_selected = param.data[:, param.
                              selected_indices] if Opt is not ZenFlowSelectiveAdamW_stage3 else param.ds_tensor.view(
                                  param.ds_shape)[param.selected_indices, :]
    diff_norm = (old_selected - new_selected).abs().sum().item()

    assert diff_norm > 1e-5, "param was not updated"
    assert param.temp_selected_param is None
    assert param.selected_grad is None


@pytest.mark.parametrize("Opt", [ZenFlowSelectiveAdamW, ZenFlowSelectiveAdamW_stage3])
def test_step_with_offload_bucket_flush(Opt):
    param1 = make_param(Opt, (2, 4), torch.tensor([1, 2]))
    param2 = make_param(Opt, (2, 4), torch.tensor([0, 3]))

    param1.exp_avg = torch.zeros_like(param1.temp_selected_param)
    param1.exp_avg_sq = torch.zeros_like(param1.temp_selected_param)
    param1.exp_avg_cpu_data = param1.exp_avg.clone().cpu()
    param1.exp_avg_sq_cpu_data = param1.exp_avg_sq.clone().cpu()

    param2.exp_avg = torch.zeros_like(param2.temp_selected_param)
    param2.exp_avg_sq = torch.zeros_like(param2.temp_selected_param)
    param2.exp_avg_cpu_data = param2.exp_avg.clone().cpu()
    param2.exp_avg_sq_cpu_data = param2.exp_avg_sq.clone().cpu()

    opt = Opt([param1, param2], lr=1e-3, offload=True, bucket_size=1)
    opt.step()
    assert param1.temp_selected_param is None
    assert param2.temp_selected_param is None


@pytest.mark.parametrize("Opt", [ZenFlowSelectiveAdamW, ZenFlowSelectiveAdamW_stage3])
def test_clear_selected_mv(Opt):
    param = make_param(Opt, (2, 4), torch.tensor([0, 2]))
    opt = Opt([param], lr=1e-3, offload=False)
    opt.step()
    state = opt.state[param]
    assert "exp_avg" in state
    opt.clear_selected_mv()
    assert state["exp_avg"].abs().sum() == 0


@pytest.mark.parametrize("Opt", [ZenFlowSelectiveAdamW, ZenFlowSelectiveAdamW_stage3])
def test_group_step_without_offload(Opt):
    param = make_param(Opt, (2, 6), torch.tensor([0, 1, 3]))
    opt = Opt([param], lr=1e-3, offload=False)
    group_to_paramlist = {0: [param]} if not Opt is ZenFlowSelectiveAdamW_stage3 else [param]
    opt.group_step(group_to_paramlist)
    assert param.selected_grad is None


@pytest.mark.parametrize("Opt", [ZenFlowSelectiveAdamW, ZenFlowSelectiveAdamW_stage3])
def test_group_step_with_offload(Opt):
    param = make_param(Opt, (2, 6), torch.tensor([0, 1, 3]))
    opt = Opt([param], lr=1e-3, offload=True)

    state = opt.state.setdefault(param, {})
    state["step"] = torch.zeros((), dtype=param.dtype, device=param.device)
    param.exp_avg = torch.zeros_like(param.data[:, param.selected_indices])
    param.exp_avg_sq = torch.zeros_like(param.data[:, param.selected_indices])
    param.exp_avg_cpu_data = param.exp_avg.clone().cpu()
    param.exp_avg_sq_cpu_data = param.exp_avg_sq.clone().cpu()

    group_to_paramlist = {0: [param]} if Opt is not ZenFlowSelectiveAdamW_stage3 else [param]
    opt.group_step(group_to_paramlist)
    assert param.selected_grad is None


@pytest.mark.parametrize("Opt", [ZenFlowSelectiveAdamW, ZenFlowSelectiveAdamW_stage3])
def test_1d_param_support(Opt):
    param = make_param(Opt, (10, ), torch.arange(10))
    opt = Opt([param], lr=1e-3, offload=False)
    opt.step()
    assert param.temp_selected_param is None
    assert param.selected_grad is None


@pytest.mark.parametrize("Opt", [ZenFlowSelectiveAdamW, ZenFlowSelectiveAdamW_stage3])
def test_state_increment(Opt):
    param = make_param(Opt, (2, 4), torch.arange(4))

    opt = Opt([param], lr=1e-3, offload=False)
    opt.step()
    step1 = opt.state[param]['step'].item()

    param.selected_grad = torch.randn(2, 4) if Opt is not ZenFlowSelectiveAdamW_stage3 else torch.randn(4, 2)
    param.temp_selected_param = param.data.clone() if Opt is not ZenFlowSelectiveAdamW_stage3 else torch.randn(4, 2)
    param.selected_indices = torch.arange(4)

    opt.step()
    step2 = opt.state[param]['step'].item()
    assert step2 == step1 + 1


def _compare_with_torch_adamw(param, zenflow_opt, atol=1e-4):
    torch_param = torch.nn.Parameter(param.detach().clone())
    torch_opt = torch.optim.AdamW([torch_param], lr=zenflow_opt.param_groups[0]['lr'])

    for _ in range(10):
        grad = torch.randn_like(param)
        param.selected_indices = torch.arange(param.shape[1])
        param.selected_grad = grad if not isinstance(zenflow_opt, ZenFlowSelectiveAdamW_stage3) else grad.T
        param.temp_selected_param = param.data.clone() if not isinstance(
            zenflow_opt, ZenFlowSelectiveAdamW_stage3) else param.ds_tensor.view(param.ds_shape).clone()

        torch_param.grad = grad.clone()

        zenflow_opt.step()
        torch_opt.step()

    if not isinstance(zenflow_opt, ZenFlowSelectiveAdamW_stage3):
        np.testing.assert_allclose(torch_param.data.cpu().numpy(),
                                   param.data.cpu().numpy(),
                                   atol=atol,
                                   err_msg="Mismatch with torch.AdamW")
    else:
        np.testing.assert_allclose(torch_param.data.cpu().numpy(),
                                   param.ds_tensor.view(param.ds_shape).T.clone().data.cpu().numpy(),
                                   atol=atol,
                                   err_msg="Mismatch with torch.AdamW")


@pytest.mark.parametrize("Opt", [ZenFlowSelectiveAdamW, ZenFlowSelectiveAdamW_stage3])
def test_against_torch_adamw(Opt):
    param = make_param(Opt, (2, 4), torch.arange(4))
    opt = Opt([param], lr=1e-3, offload=False)
    _compare_with_torch_adamw(param, opt)
