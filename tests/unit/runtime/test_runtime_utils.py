# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import torch
from torch._utils import _flatten_dense_tensors
import deepspeed.comm as dist
import pytest
from typing import Dict

import deepspeed
import deepspeed.runtime.utils as ds_utils
import deepspeed.utils.groups as groups
from deepspeed.accelerator import get_accelerator
from deepspeed.moe.layer import MoE
from deepspeed.moe.utils import is_moe_param, split_params_into_different_moe_groups_for_optimizer

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

    def test_sharded_experts_count_once(self):
        """The global norm must not depend on how experts are spread over ranks.

        Rank-averaging the per-rank norms only reconstructs a global norm when every
        rank holds the same parameters. Under expert parallelism they do not, so the
        four experts below gave sqrt(20) and sqrt(100) averaged to 7.236 instead of
        the sqrt(120) = 10.954 that counting each expert once produces (#8469).
        """
        groups._create_expert_and_data_parallel(2)
        rank = dist.get_rank()
        device = get_accelerator().device_name(rank)

        experts = [torch.full((4, ), float(v)) for v in (1.0, 2.0, 3.0, 4.0)]
        expected = torch.cat(experts).norm(2).item()

        owned = experts[:2] if rank == 0 else experts[2:]
        params = []
        for grad in owned:
            param = torch.nn.Parameter(torch.zeros(4, device=device))
            param.grad = grad.clone().to(device)
            param.allreduce = False
            param.group_name = "ep_size_2"
            params.append(param)

        # A max_norm far above the norm leaves the gradients alone, so this reads the
        # computed norm rather than the clipped result.
        norm = ds_utils.clip_grad_norm_(params, max_norm=1e9)

        assert abs(float(norm) -
                   expected) < 1e-4, (f"global norm {float(norm)} should be {expected} regardless of expert placement")

    def _expert(self, value, device, with_grad=True):
        param = torch.nn.Parameter(torch.zeros(4, device=device))
        if with_grad:
            param.grad = torch.full((4, ), value, device=device)
        param.allreduce = False
        param.group_name = "ep_size_2"
        return param

    def _plain(self, value, device):
        param = torch.nn.Parameter(torch.zeros(4, device=device))
        param.grad = torch.full((4, ), value, device=device)
        return param

    def test_experts_idle_on_one_rank_do_not_deadlock(self):
        """A rank whose experts got no tokens must take the same collectives (#8469).

        Ownership has to be read from the parameters, not the gradients: `p.grad is
        not None` filtering drops an unused expert, and branching on what is left made
        one rank reduce over the expert group while its peer reduced over the data
        parallel group. They then waited on each other.
        """
        groups._create_expert_and_data_parallel(2)
        rank = dist.get_rank()
        device = get_accelerator().device_name(rank)

        expert = self._expert(2.0, device, with_grad=(rank == 0))
        norm = ds_utils.clip_grad_norm_([self._plain(1.0, device), expert], max_norm=1e9)

        # non-expert 2.0 replicated, one expert of norm 4.0 owned by rank 0:
        # sqrt(2**2 + 4**2).
        assert abs(float(norm) - 20**0.5) < 1e-4

    def test_a_rank_owning_no_expert_does_not_deadlock(self):
        """Under pipeline parallelism a rank can hold no expert at all.

        Local ownership then differs in kind rather than in degree, so the branch is
        settled by one all-reduced flag instead of by what this rank happens to hold.
        """
        groups._create_expert_and_data_parallel(2)
        rank = dist.get_rank()
        device = get_accelerator().device_name(rank)

        params = [self._plain(1.0, device)]
        if rank == 0:
            params.append(self._expert(2.0, device))

        norm = ds_utils.clip_grad_norm_(params, max_norm=1e9)
        assert abs(float(norm) - 20**0.5) < 1e-4

    def test_expert_only_inf_norm(self):
        """Nothing replicated leaves no tensor to take a max over; zero is the identity."""
        groups._create_expert_and_data_parallel(2)
        rank = dist.get_rank()
        device = get_accelerator().device_name(rank)

        norm = ds_utils.clip_grad_norm_([self._expert(float(rank + 1), device)], max_norm=1e9, norm_type=float('inf'))

        assert abs(float(norm) - 2.0) < 1e-4

    def test_a_rank_with_no_gradient_at_all_still_gets_the_norm(self):
        """Every local gradient can be absent, expert ones included, and the rank still has to answer."""
        groups._create_expert_and_data_parallel(2)
        rank = dist.get_rank()
        device = get_accelerator().device_name(rank)

        norm = ds_utils.clip_grad_norm_([self._expert(2.0, device, with_grad=(rank == 1))], max_norm=1e9)

        # Only rank 1's expert has a gradient, of norm 4.0, and both ranks report it.
        assert abs(float(norm) - 4.0) < 1e-4

    def test_expert_groups_are_visited_in_registry_order(self, monkeypatch):
        """Ranks owning different expert groups must still walk the groups in one order."""
        groups._create_expert_and_data_parallel(2)
        groups._create_expert_and_data_parallel(1)
        rank = dist.get_rank()
        device = get_accelerator().device_name(rank)

        visited = []
        real = ds_utils.get_norm_with_moe_layers

        def spy(non_expert_norm, mpu, expert_tensors, norm_type=2):
            visited.append(list(expert_tensors))
            return real(non_expert_norm, mpu=mpu, expert_tensors=expert_tensors, norm_type=norm_type)

        monkeypatch.setattr(ds_utils, "get_norm_with_moe_layers", spy)

        expert = self._expert(2.0, device)
        expert.group_name = "ep_size_2" if rank == 0 else "ep_size_1"
        ds_utils.clip_grad_norm_([expert], max_norm=1e9)

        assert visited == [["ep_size_1", "ep_size_2"]]

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


class _OneMoELayer(torch.nn.Module):
    """A dense layer, then one expert per rank behind a gate."""

    def __init__(self, hidden_dim):
        super().__init__()
        self.dense = torch.nn.Linear(hidden_dim, hidden_dim)
        expert = torch.nn.Linear(hidden_dim, hidden_dim)
        # k=2 sends every token to both experts, so each rank's expert gets a gradient.
        self.moe = MoE(hidden_size=hidden_dim, expert=expert, num_experts=2, ep_size=2, k=2, capacity_factor=2.0)

    def forward(self, x):
        output, _, _ = self.moe(self.dense(x))
        return output.pow(2).mean()


class TestClipFp32GradientsThroughEngine(DistributedTest):
    """The clip `DeepSpeedEngine.step()` applies in an fp32 run without ZeRO, with experts on two ranks.

    `clip_grad_norm_` returns its norm and the engine drops it, so the norm is read from its effect. With
    plain SGD the update is `lr * clip_coef * grad`, and the coefficient has to come from the norm of every
    gradient with each expert counted once, not from the ranks' own norms averaged.
    """

    world_size = 2

    def test_the_update_is_clipped_by_the_global_norm(self):
        hidden_dim, max_norm, lr = 8, 1e-3, 1.0
        torch.manual_seed(0)
        model = _OneMoELayer(hidden_dim)
        param_group = {'params': list(model.parameters()), 'name': 'all'}
        optimizer = torch.optim.SGD(split_params_into_different_moe_groups_for_optimizer(param_group), lr=lr)
        config = {
            "train_micro_batch_size_per_gpu": 2,
            "gradient_clipping": max_norm,
            "zero_optimization": {
                "stage": 0
            }
        }
        engine, _, _, _ = deepspeed.initialize(config=config, model=model, optimizer=optimizer)

        # A different loss scale on each rank gives the two experts gradients of different size.
        rank = dist.get_rank()
        generator = torch.Generator().manual_seed(100 + rank)
        x = torch.randn(2, 4, hidden_dim, generator=generator).to(engine.device)
        engine.backward(engine(x) * (1 + 3 * rank))

        params = list(engine.module.parameters())
        dense = [p for p in params if not is_moe_param(p)]
        experts = [p for p in params if is_moe_param(p)]
        assert dense and experts
        dense_sq = torch.stack([p.grad.float().pow(2).sum() for p in dense]).sum()
        expert_sq = torch.stack([p.grad.float().pow(2).sum() for p in experts]).sum()

        # The oracle counts a dense gradient once because it is the same on both ranks, and an expert once
        # because each rank owns a different one.
        dense_on_every_rank = dense_sq.clone()
        dist.all_reduce(dense_on_every_rank, op=dist.ReduceOp.MIN)
        assert torch.equal(dense_on_every_rank, dense_sq), "dense gradients should match across ranks"
        all_experts_sq = expert_sq.clone()
        dist.all_reduce(all_experts_sq)
        true_norm = (dense_sq + all_experts_sq).sqrt().item()

        # Averaging the ranks' own norms is the answer this test exists to rule out, so it has to differ.
        own_norm = (dense_sq + expert_sq).sqrt()
        dist.all_reduce(own_norm)
        averaged_norm = own_norm.item() / dist.get_world_size()
        assert true_norm > 5 * max_norm, "the clip must be active for this to say anything"
        assert abs(averaged_norm - true_norm) > 0.05 * true_norm, "the two norms must be distinguishable"

        clip_coef = max_norm / (true_norm + 1e-6)
        before = [p.detach().clone() for p in params]
        grads = [p.grad.detach().clone() for p in params]
        engine.step()

        for p, start, grad in zip(params, before, grads):
            torch.testing.assert_close(p.detach(), start - lr * clip_coef * grad, rtol=1e-4, atol=1e-6)


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
