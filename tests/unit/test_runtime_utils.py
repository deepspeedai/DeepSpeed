from deepspeed.moe.utils import is_moe_param, split_params_grads_into_shared_and_expert_params, split_params_into_shared_and_expert_params
import torch
from torch._utils import _flatten_dense_tensors
import torch.distributed as dist
import pytest

import deepspeed.runtime.utils as ds_utils
from deepspeed.utils.logging import log_dist
import deepspeed.utils.groups as groups

from .common import distributed_test


def test_call_to_str():
    c2s = ds_utils.call_to_str

    assert c2s('int') == 'int()'
    assert c2s('int', 3) == 'int(3)'
    assert c2s('int', 3, 'jeff') == 'int(3, \'jeff\')'

    assert c2s('hello', val=3) == 'hello(val=3)'
    assert c2s('hello', 1138, val=3) == 'hello(1138, val=3)'


def test_clip_grad_norm_():
    @distributed_test(world_size=[2])
    def _test_clip_grad_norm_() -> None:
        param1 = torch.nn.Parameter(torch.Tensor([0]))
        param1.grad = torch.Tensor([1])
        param2 = torch.nn.Parameter(torch.Tensor([0]))
        param2.grad = torch.Tensor([dist.get_rank() + 1])
        # param2 is now MoE parameter
        param2.allreduce = False

        parameters = [param1, param2]

        groups.initialize_model_parallel(1)
        groups.initialize_expert_parallel(2)

        norm = ds_utils.clip_grad_norm_(parameters, max_norm=0.1)
        norm = torch.Tensor([norm]).to(dist.get_rank())

        world_size = dist.get_world_size()
        gathered_norm = [torch.zeros(1).cuda() for i in range(world_size)]

        torch.distributed.all_gather(gathered_norm, norm)

        assert gathered_norm[0] == gathered_norm[1], "norm at rank 0 does not match the norm at rank 1"

    return _test_clip_grad_norm_()


@pytest.mark.parametrize("check_using_norm", [(False), (True)])
def test_CheckOverflow(check_using_norm):
    @distributed_test(world_size=[2])
    def _test_CheckOverflow(check_using_norm: bool):
        groups.initialize_model_parallel(1)
        groups.initialize_expert_parallel(2)

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

    return _test_CheckOverflow(check_using_norm)
