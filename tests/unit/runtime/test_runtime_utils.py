# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import torch
import deepspeed
from torch._utils import _flatten_dense_tensors
import deepspeed.comm as dist
import pytest
from typing import Dict

import deepspeed.runtime.utils as ds_utils
import deepspeed.utils.groups as groups
from deepspeed.accelerator import get_accelerator
from deepspeed.runtime.bf16_optimizer import BF16_Optimizer

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


class TestClipGradNormPNorm(DistributedTest):
    # world_size 1 so this runs wherever the suite runs; the bug is in the per-rank
    # recombination of the norms, which is independent of the group size.
    world_size = 1

    @pytest.mark.parametrize("norm_type", [1, 2, 3])
    def test_matches_torch(self, norm_type):
        # The p-norm over all gradients is (sum_i ||g_i||_p ** p) ** (1/p). Squaring the
        # per-parameter norms computes that only for p == 2, which is the control here.
        def test_params():
            param1 = torch.nn.Parameter(torch.zeros(2))
            param1.grad = torch.Tensor([3.0, -4.0])
            param2 = torch.nn.Parameter(torch.zeros(1))
            param2.grad = torch.Tensor([2.0])
            return [param1, param2]

        max_norm = 1.0
        params_expected = test_params()
        expected_norm = torch.nn.utils.clip_grad_norm_(params_expected, max_norm, norm_type=norm_type)

        params_actual = test_params()
        actual_norm = ds_utils.clip_grad_norm_(params_actual, max_norm=max_norm, norm_type=norm_type)

        assert torch.allclose(actual_norm.float().cpu(), expected_norm.float().cpu())
        for expected, actual in zip(params_expected, params_actual):
            assert torch.allclose(actual.grad, expected.grad)


class TestGlobalNormMoEGroup(DistributedTest):
    world_size = 2

    @pytest.mark.parametrize("norm_type", [2, float("inf")])
    @pytest.mark.parametrize("kind", ["pair", "single", "none"])
    def test(self, norm_type, kind):
        rank = dist.get_rank()
        if kind == "none":
            moe_ep_group = None
        elif kind == "pair":
            moe_ep_group = dist.new_group([0, 1])
        else:
            moe_ep_group = [dist.new_group([r]) for r in range(self.world_size)][rank]
        local = 3.0 if rank == 0 else 4.0
        grads = [torch.tensor([local], device=get_accelerator().device_name())]

        norm = ds_utils.get_global_norm_of_tensors(grads, norm_type=norm_type, moe_ep_group=moe_ep_group)

        # Expert norms combine across the expert-parallel group only: 3-4-5 for the 2-norm, 4 for the
        # max norm. Without expert parallelism, None must not reduce over the world group.
        if kind == "pair":
            expected = 5.0 if norm_type == 2 else 4.0
        else:
            expected = local
        assert norm.item() == pytest.approx(expected)


class _NormModelParallelUnit:

    def __init__(self, model_parallel_group, data_parallel_group=None):
        self.model_parallel_group = model_parallel_group
        self.data_parallel_group = data_parallel_group

    def get_model_parallel_group(self):
        return self.model_parallel_group

    def get_model_parallel_rank(self):
        return dist.get_rank(self.model_parallel_group)

    def get_model_parallel_world_size(self):
        return dist.get_world_size(self.model_parallel_group)

    def get_data_parallel_group(self):
        return self.data_parallel_group

    def get_data_parallel_rank(self):
        return dist.get_rank(self.data_parallel_group)

    def get_data_parallel_world_size(self):
        return dist.get_world_size(self.data_parallel_group)


class TestGlobalNormEmptyInput(DistributedTest):
    world_size = 2

    @pytest.mark.parametrize("parallelism", ["model", "expert", "none"])
    @pytest.mark.parametrize("input_case", ["nonempty", "one_empty", "all_empty"])
    def test(self, parallelism, input_case):
        rank = dist.get_rank()
        parallel_group = dist.new_group([0, 1])
        kwargs = {}
        if parallelism == "model":
            kwargs["mpu"] = _NormModelParallelUnit(parallel_group)
        elif parallelism == "expert":
            kwargs["moe_ep_group"] = parallel_group

        local_values = [3.0, -4.0] if rank == 0 else [12.0]
        if input_case == "all_empty" or (input_case == "one_empty" and rank == 1):
            local_values = []
        tensors = [torch.tensor(local_values, device=get_accelerator().current_device_name())] if local_values else []
        expected_norms = [0.0, 0.0, 0.0, 0.0]
        if local_values:
            expected_norms = [7.0, 5.0, 91.0**(1.0 / 3.0), 4.0] if rank == 0 else [12.0] * 4
        if parallelism != "none" and input_case != "all_empty":
            expected_norms = ([19.0, 13.0, 1819.0**(1.0 / 3.0), 12.0]
                              if input_case == "nonempty" else [7.0, 5.0, 91.0**(1.0 / 3.0), 4.0])

        for norm_type, expected in zip([1, 2, 3, float("inf")], expected_norms):
            norm = ds_utils.get_global_norm_of_tensors(tensors, norm_type=norm_type, **kwargs)
            assert norm.item() == pytest.approx(expected)
            assert norm.dtype == torch.float32
            assert norm.ndim == 0
            assert norm.device == torch.device(get_accelerator().current_device_name())
            assert not norm.requires_grad


class TestGlobalNormIterable(DistributedTest):
    world_size = 1

    @pytest.mark.parametrize("input_case", ["nonempty", "empty"])
    def test(self, input_case):
        for device in ["cpu", get_accelerator().current_device_name()]:
            tensors = [torch.tensor([3.0, -4.0], device=device, requires_grad=True)
                       ] if input_case == "nonempty" else []
            for norm_type, nonempty_expected in [(1, 7.0), (2, 5.0), (3, 91.0**(1.0 / 3.0)), (float("inf"), 4.0)]:
                norm = ds_utils.get_global_norm_of_tensors((tensor for tensor in tensors), norm_type=norm_type)
                assert norm.item() == pytest.approx(nonempty_expected if tensors else 0.0)
                expected_device = tensors[0].device if tensors else torch.device(
                    get_accelerator().current_device_name())
                assert norm.device == expected_device
                assert norm.dtype == torch.float32
                assert norm.ndim == 0
                assert not norm.requires_grad


class TestBF16EmptyNormRank(DistributedTest):
    world_size = 2

    def test(self):
        if torch.bfloat16 not in get_accelerator().supported_dtypes():
            pytest.skip("requires bfloat16 support")
        model_parallel_group = dist.new_group([0, 1])
        data_parallel_group = [dist.new_group([rank]) for rank in range(self.world_size)][dist.get_rank()]
        mpu = _NormModelParallelUnit(model_parallel_group, data_parallel_group)
        model = torch.nn.Linear(2, 1, bias=False)
        torch.nn.init.ones_(model.weight)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        config = {
            "train_micro_batch_size_per_gpu": 1,
            "bf16": {
                "enabled": True
            },
            "data_types": {
                "grad_accum_dtype": "fp32"
            },
            "zero_optimization": {
                "stage": 1
            },
            "zero_allow_untested_optimizer": True,
            "gradient_clipping": 1.0,
        }
        engine, _, _, _ = deepspeed.initialize(model=model, optimizer=optimizer, config=config, mpu=mpu)
        try:
            assert isinstance(engine.optimizer, BF16_Optimizer)
            inputs = torch.tensor([[3.0, 4.0]], device=engine.device, dtype=torch.bfloat16)
            engine.backward(engine(inputs).sum())
            # Replicated parameters contribute to the norm only on tensor-parallel rank zero.
            norm_grads, _ = engine.optimizer.get_grads_for_norm()
            assert len(norm_grads) == (1 if dist.get_rank() == 0 else 0)
            engine.step()
            assert engine.get_global_grad_norm().item() == pytest.approx(5.0)
            expected = torch.tensor([[0.94, 0.92]], device=engine.device, dtype=torch.bfloat16)
            torch.testing.assert_close(engine.module.weight, expected, rtol=0, atol=0)
        finally:
            engine.destroy()


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


@pytest.mark.skipif(not hasattr(torch.autograd.graph, "_get_grad_fn_or_grad_acc"),
                    reason="requires torch.autograd.graph._get_grad_fn_or_grad_acc")
def test_count_used_parameters_enables_grad_for_grad_acc_lookup(monkeypatch):
    """count_used_parameters_in_backward should enable grad for grad-acc lookup."""
    param = torch.nn.Parameter(torch.tensor([1.0], requires_grad=True))
    seen: Dict[str, int] = {"lookup_calls": 0}
    original_getter = torch.autograd.graph._get_grad_fn_or_grad_acc

    def _require_grad_enabled(t):
        seen["lookup_calls"] += 1
        if not torch.is_grad_enabled():
            raise RuntimeError("grad mode must be enabled for grad-acc lookup")
        return original_getter(t)

    monkeypatch.setattr(torch.autograd.graph, "_get_grad_fn_or_grad_acc", _require_grad_enabled)

    def _hook(grad):
        seen["count"] = ds_utils.count_used_parameters_in_backward([param])
        return grad

    param.register_hook(_hook)
    loss = (param * 2.0).sum()
    loss.backward()
    assert seen["lookup_calls"] > 0
    assert "count" in seen
