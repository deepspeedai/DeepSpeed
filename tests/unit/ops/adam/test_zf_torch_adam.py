# Copyright (c) DeepSpeed Team.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import torch
import numpy as np
from torch.nn import Parameter
from deepspeed.ops.adam import ZenFlowSelectiveAdamW


def make_param(shape, selected_indices=None):
    param = Parameter(torch.randn(*shape))
    if selected_indices is not None:
        param.selected_indices = selected_indices
        param.selected_grad = torch.randn(param.shape[0], len(selected_indices))
        param.temp_selected_param = param.data[:, selected_indices].clone()
    return param


def test_init_methods():
    opt1 = ZenFlowSelectiveAdamW([torch.nn.Parameter(torch.randn(2, 4))], lr=1e-3, offload=False)
    assert opt1.step == opt1._step_without_offload
    assert opt1.group_step == opt1._group_step_without_offload
    opt2 = ZenFlowSelectiveAdamW([torch.nn.Parameter(torch.randn(2, 4))], lr=1e-3, offload=True)
    assert opt2.step == opt2._step_with_offload
    assert opt2.group_step == opt2._group_step_with_offload


def test_step_without_offload():
    param = make_param((4, 6), torch.tensor([1, 3, 4]))
    param.requires_grad_(True)
    opt = ZenFlowSelectiveAdamW([param], lr=1e-3, offload=False)

    old_selected = param.data[:, param.selected_indices].clone()

    opt.step()

    new_selected = param.data[:, param.selected_indices]
    diff_norm = (old_selected - new_selected).abs().sum().item()

    assert diff_norm > 1e-5, "param was not updated"
    assert param.temp_selected_param is None
    assert param.selected_grad is None


def test_step_with_offload_bucket_flush():
    param1 = make_param((2, 4), torch.tensor([1, 2]))
    param2 = make_param((2, 4), torch.tensor([0, 3]))

    param1.exp_avg = torch.zeros_like(param1.temp_selected_param)
    param1.exp_avg_sq = torch.zeros_like(param1.temp_selected_param)
    param1.exp_avg_cpu_data = param1.exp_avg.clone().cpu()
    param1.exp_avg_sq_cpu_data = param1.exp_avg_sq.clone().cpu()

    param2.exp_avg = torch.zeros_like(param2.temp_selected_param)
    param2.exp_avg_sq = torch.zeros_like(param2.temp_selected_param)
    param2.exp_avg_cpu_data = param2.exp_avg.clone().cpu()
    param2.exp_avg_sq_cpu_data = param2.exp_avg_sq.clone().cpu()

    opt = ZenFlowSelectiveAdamW([param1, param2], lr=1e-3, offload=True, bucket_size=1)
    opt.step()
    assert param1.temp_selected_param is None
    assert param2.temp_selected_param is None


def test_clear_selected_mv():
    param = make_param((2, 4), torch.tensor([0, 2]))
    opt = ZenFlowSelectiveAdamW([param], lr=1e-3, offload=False)
    opt.step()
    state = opt.state[param]
    assert "exp_avg" in state
    opt.clear_selected_mv()
    assert state["exp_avg"].abs().sum() == 0


def test_group_step_without_offload():
    param = make_param((2, 6), torch.tensor([0, 1, 3]))
    opt = ZenFlowSelectiveAdamW([param], lr=1e-3, offload=False)
    group_to_paramlist = {0: [param]}
    opt._group_step_without_offload(group_to_paramlist)
    assert param.selected_grad is None


def test_group_step_with_offload():
    param = make_param((2, 6), torch.tensor([0, 1, 3]))
    opt = ZenFlowSelectiveAdamW([param], lr=1e-3, offload=True)

    state = opt.state.setdefault(param, {})
    state["step"] = torch.zeros((), dtype=param.dtype, device=param.device)
    param.exp_avg = torch.zeros_like(param.data[:, param.selected_indices])
    param.exp_avg_sq = torch.zeros_like(param.data[:, param.selected_indices])
    param.exp_avg_cpu_data = param.exp_avg.clone().cpu()
    param.exp_avg_sq_cpu_data = param.exp_avg_sq.clone().cpu()

    group_to_paramlist = {0: [param]}
    opt._group_step_with_offload(group_to_paramlist)
    assert param.selected_grad is None


def test_1d_param_support():
    param = Parameter(torch.randn(10))
    param.selected_grad = torch.randn(10)
    param.temp_selected_param = param.data.clone()
    opt = ZenFlowSelectiveAdamW([param], lr=1e-3, offload=False)
    opt.step()
    assert param.temp_selected_param is None
    assert param.selected_grad is None


def test_state_increment():
    param = torch.nn.Parameter(torch.randn(2, 4))
    param.selected_indices = torch.arange(4)
    param.selected_grad = torch.randn(2, 4)
    param.temp_selected_param = param.data.clone()

    opt = ZenFlowSelectiveAdamW([param], lr=1e-3, offload=False)
    opt.step()
    step1 = opt.state[param]['step'].item()

    param.selected_grad = torch.randn(2, 4)
    param.temp_selected_param = param.data.clone()
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
        param.selected_grad = grad
        param.temp_selected_param = param.data.clone()

        torch_param.grad = grad.clone()

        zenflow_opt.step()
        torch_opt.step()

    np.testing.assert_allclose(torch_param.data.cpu().numpy(),
                               param.data.cpu().numpy(),
                               atol=atol,
                               err_msg="Mismatch with torch.AdamW")


def test_against_torch_adamw():
    param = torch.nn.Parameter(torch.randn(2, 4))
    param.selected_indices = torch.arange(4)
    opt = ZenFlowSelectiveAdamW([param], lr=1e-3, offload=False)
    _compare_with_torch_adamw(param, opt)
