from deepspeed.moe.utils import is_moe_param, split_params_grads_into_shared_and_expert_params, split_params_into_shared_and_expert_params
import torch
from torch._utils import _flatten_dense_tensors
import torch.distributed as dist
import pytest

import deepspeed.runtime.utils as ds_utils
from deepspeed.utils.logging import log_dist
import deepspeed.utils.groups as groups

from common import distributed_test


def test_call_to_str():
    c2s = ds_utils.call_to_str

    assert c2s('int') == 'int()'
    assert c2s('int', 3) == 'int(3)'
    assert c2s('int', 3, 'jeff') == 'int(3, \'jeff\')'

    assert c2s('hello', val=3) == 'hello(val=3)'
    assert c2s('hello', 1138, val=3) == 'hello(1138, val=3)'


@pytest.mark.parametrize('ignore_expert_params', [(False), (True)])
def test_clip_grad_norm_(ignore_expert_params: bool):
    @distributed_test(world_size=[2])
    def _test_clip_grad_norm_(ignore_expert_params: bool) -> None:
        param1 = torch.nn.Parameter(torch.Tensor([0]))
        param1.grad = torch.Tensor([1])
        param2 = torch.nn.Parameter(torch.Tensor([0]))
        param2.grad = torch.Tensor([dist.get_rank() + 1])

        param2.allreduce = False
        # param2 is now MoE parameter
        parameters = [param1, param2]
        if not ignore_expert_params:
            groups.initialize_model_parallel(1)
            groups.initialize_expert_parallel(2)
        norm = ds_utils.clip_grad_norm_(parameters,
                                        max_norm=0.1,
                                        ignore_expert_params=ignore_expert_params)
        if ignore_expert_params:
            # Ignore param2.grad
            assert norm == 1.0
        else:
            # Use param2.grad from both ranks
            assert torch.isclose(torch.Tensor([norm]), torch.sqrt(torch.Tensor([6])))

    return _test_clip_grad_norm_(ignore_expert_params)


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
