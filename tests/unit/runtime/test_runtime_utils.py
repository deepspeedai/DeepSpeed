# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import torch
from torch._utils import _flatten_dense_tensors
import deepspeed.comm as dist
import pytest

import deepspeed.runtime.utils as ds_utils
import deepspeed.utils.groups as groups
from deepspeed.accelerator import get_accelerator

from unit.common import DistributedTest


def test_call_to_str():
    c2s = ds_utils.call_to_str

    assert c2s('int') == 'int()'
    assert c2s('int', 3) == 'int(3)'
    assert c2s('int', 3, 'jeff') == 'int(3, \'jeff\')'

    assert c2s('hello', val=3) == 'hello(val=3)'
    assert c2s('hello', 1138, val=3) == 'hello(1138, val=3)'


class TestClipGradNorm(DistributedTest):
    world_size = 2

    def test_gather(self):
        param1 = torch.nn.Parameter(torch.Tensor([0]))
        param1.grad = torch.Tensor([1])
        param2 = torch.nn.Parameter(torch.Tensor([0]))
        param2.grad = torch.Tensor([dist.get_rank() + 1])
        # param2 is now MoE parameter
        param2.allreduce = False

        parameters = [param1, param2]

        groups._create_expert_and_data_parallel(2)

        norm = ds_utils.clip_grad_norm_(parameters, max_norm=0.1)
        norm = torch.Tensor([norm]).to(get_accelerator().device_name(dist.get_rank()))
        world_size = dist.get_world_size()
        gathered_norm = [torch.zeros(1).to(get_accelerator().device_name()) for i in range(world_size)]

        dist.all_gather(gathered_norm, norm)

        assert gathered_norm[0] == gathered_norm[1], "norm at rank 0 does not match the norm at rank 1"

    def test_clipped_val(self):
        max_norm = 0.1

        def test_params():
            param1 = torch.nn.Parameter(torch.Tensor([0]))
            param1.grad = torch.Tensor([1])
            param2 = torch.nn.Parameter(torch.Tensor([0]))
            param2.grad = torch.Tensor([1])
            return [param1, param2]

        # This assumes gradients are same on all the ranks and doesn't consider multiple ranks
        params_expected = test_params()
        torch.nn.utils.clip_grad_norm_(params_expected, max_norm)

        params_actual = test_params()
        ds_utils.clip_grad_norm_(params_actual, max_norm=max_norm)

        # This can be allclose
        assert torch.equal(params_expected[0].grad, params_actual[0].grad)
        assert torch.equal(params_expected[1].grad, params_actual[1].grad)


@pytest.mark.parametrize("check_using_norm", [(False), (True)])
class TestCheckOverflow(DistributedTest):
    world_size = 2

    def test(self, check_using_norm):
        groups._create_expert_and_data_parallel(2)

        param1 = torch.nn.Parameter(torch.Tensor([0]))
        param1.grad = torch.Tensor([1])
        param2 = torch.nn.Parameter(torch.Tensor([0]))
        if dist.get_rank() == 0:
            param2.grad = torch.Tensor([1])
        else:
            param2.grad = torch.Tensor([float("inf")])
        param2.allreduce = False
        # param2 is now MoE parameter
        parameters = [param1, param2]
        if check_using_norm:
            grads_group_flat = [_flatten_dense_tensors([p.grad for p in parameters])]
            norm = ds_utils.get_weight_norm(grads_group_flat)
            overflow_checker = ds_utils.CheckOverflow([parameters])
            overflow = overflow_checker.check_using_norm([norm], reduce_overflow=False)
        else:
            overflow_checker = ds_utils.CheckOverflow([parameters])
            overflow = overflow_checker.check()
        assert overflow
