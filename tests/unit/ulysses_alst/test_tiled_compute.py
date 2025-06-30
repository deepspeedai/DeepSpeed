# Copyright (c) The DeepSpeed Contributors
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
"""
Arctic Long Sequence Training (ALST) Tiled compute component tests
"""

from deepspeed.runtime.sequence_parallel.ulysses_sp import TiledMLP, sequence_tiled_compute
from deepspeed.utils import safe_get_full_grad
from torch.nn import Linear, Module
from unit.common import DistributedTest, preferred_dtype
from unit.util import torch_assert_equal, torch_assert_close
import deepspeed
import pytest
import torch


def get_grad(param, zero_stage):
    return safe_get_full_grad(param)
    # z1 now has contiguous_gradients enabled by default so `param.grad is None` even under z1
    # if zero_stage == 1:
    #     return param.grad
    # else:
    #     return safe_get_full_grad(param)


class SimpleMLP(Module):

    def __init__(self, hidden_dim):
        super().__init__()
        self.up_proj = Linear(hidden_dim, hidden_dim * 2, bias=False)
        self.down_proj = Linear(hidden_dim * 2, hidden_dim, bias=False)
        self.act = torch.nn.ReLU()

    def forward(self, x):
        return self.down_proj(self.act(self.up_proj(x)))


# save the original implementation to pass through to the tiled computation wrapper
mlp_forward_orig = SimpleMLP.forward


class MyModel(Module):

    def __init__(self, hidden_dim):
        super().__init__()
        # Critical - need to use a stack of at least 2 mlps to validate that the backward of the last mlp sends the correct gradients to the previous mlp in the stack
        self.mlp1 = SimpleMLP(hidden_dim)
        self.mlp2 = SimpleMLP(hidden_dim)
        self.cross_entropy_loss = torch.nn.CrossEntropyLoss()

    def forward(self, x, y):
        x = self.mlp1(x)
        x = self.mlp2(x)
        return self.cross_entropy_loss(x, y)


def mlp_forward_tiled_mlp(self, x):
    # this tests TiledMLP
    compute_params = [self.down_proj.weight, self.up_proj.weight]
    num_shards = 4

    return TiledMLP.apply(
        mlp_forward_orig,
        self,
        x,
        num_shards,
        compute_params,
    )


def mlp_forward_sequence_tiled_compute(self, x):
    # this tests: sequence_tiled_compute + SequenceTiledCompute - same as TiledMLP but a-non-MLP
    # specific generic implementation of tiled compute

    kwargs_to_shard = dict(x=x)
    kwargs_to_pass = dict(self=self)
    grad_requiring_tensor_key = "x"
    compute_params = [self.down_proj.weight, self.up_proj.weight]
    seqlen = x.shape[1]
    num_shards = 4

    return sequence_tiled_compute(
        mlp_forward_orig,
        seqlen,
        num_shards,
        kwargs_to_shard,
        kwargs_to_pass,
        grad_requiring_tensor_key,
        compute_params,
        output_unshard_dimension=1,  # x
        output_reduction=None,
    )


@pytest.mark.parametrize("zero_stage", [1, 3])
class TestTiledCompute(DistributedTest):
    world_size = 1

    def test_tiled_mlp(self, zero_stage):

        config_dict = {
            "train_micro_batch_size_per_gpu": 1,
            "zero_optimization": {
                "stage": zero_stage
            },
            "optimizer": {
                "type": "Adam",
                "params": {
                    "lr": 1e-3
                }
            },
        }
        dtype = preferred_dtype()
        if dtype == torch.bfloat16:
            config_dict["bf16"] = {"enabled": True}
        elif dtype == torch.float16:
            config_dict["fp16"] = {"enabled": True, "loss_scale": 1.0}

        # for debug
        # torch.set_printoptions(precision=8, sci_mode=True)

        seed = 42
        hidden_dim = 100
        bs = 1
        seqlen = hidden_dim
        torch.manual_seed(seed)
        x = torch.rand((bs, seqlen, hidden_dim), dtype=dtype, requires_grad=True)
        y = torch.empty((bs, seqlen), dtype=torch.long, requires_grad=False).random_(hidden_dim)

        # A. Baseline: model with normal MLP
        torch.manual_seed(seed)
        model_a = MyModel(hidden_dim=hidden_dim).to(dtype)
        model_a, _, _, _ = deepspeed.initialize(config=config_dict,
                                                model=model_a,
                                                model_parameters=model_a.parameters())

        x = x.to(model_a.device)
        y = y.to(model_a.device)

        x_a = x.clone().detach().requires_grad_(True)
        y_a = y.clone().detach()

        loss_a = model_a(x_a, y_a)
        model_a.backward(loss_a)
        grad_a1 = get_grad(model_a.module.mlp1.up_proj.weight, zero_stage)
        grad_a2 = get_grad(model_a.module.mlp2.up_proj.weight, zero_stage)
        assert grad_a1 is not None
        assert grad_a2 is not None

        # B. model with tiled MLP using TiledMLP
        torch.manual_seed(seed)
        SimpleMLP.forward = mlp_forward_tiled_mlp
        model_b = MyModel(hidden_dim=hidden_dim).to(dtype)
        model_b, _, _, _ = deepspeed.initialize(config=config_dict,
                                                model=model_b,
                                                model_parameters=model_b.parameters())

        x_b = x.clone().detach().requires_grad_(True)
        y_b = y.clone().detach()
        loss_b = model_b(x_b, y_b)
        model_b.backward(loss_b)
        grad_b1 = get_grad(model_b.module.mlp1.up_proj.weight, zero_stage)
        grad_b2 = get_grad(model_b.module.mlp2.up_proj.weight, zero_stage)
        assert grad_b1 is not None
        assert grad_b2 is not None

        # print(f"{loss_a=}")
        # print(f"{loss_b=}")
        # print(f"{grad_a1=}")
        # print(f"{grad_b1=}")
        # print(f"{grad_a2=}")
        # print(f"{grad_b2=}")
        torch_assert_equal(loss_a, loss_b)

        # Gradient will not be exactly the same, especially under half-precision. And bf16 is
        # particularly lossy so need to lower tolerance a bit more than the default. Switch to
        # dtype torch.float or even torch.double to see that the diff is tiny - so the math is
        # correct, but accumulation error adds up. Alternatively making hidden_dim bigger makes the
        # divergence much smaller as well.
        torch_assert_close(grad_a1, grad_b1)  #, rtol=1e-03, atol=1e-04)
        torch_assert_close(grad_a2, grad_b2)  #, rtol=1e-03, atol=1e-04)

        # C. model with tiled MLP using the generic version of the same via sequence_tiled_compute + SequenceTiledCompute
        torch.manual_seed(seed)
        SimpleMLP.forward = mlp_forward_sequence_tiled_compute
        model_c = MyModel(hidden_dim=hidden_dim).to(dtype)
        model_c, _, _, _ = deepspeed.initialize(config=config_dict,
                                                model=model_c,
                                                model_parameters=model_c.parameters())

        x_c = x.clone().detach().requires_grad_(True)
        y_c = y.clone().detach()
        loss_c = model_c(x_c, y_c)
        model_c.backward(loss_c)
        grad_c1 = get_grad(model_c.module.mlp1.up_proj.weight, zero_stage)
        grad_c2 = get_grad(model_c.module.mlp2.up_proj.weight, zero_stage)
        assert grad_c1 is not None
        assert grad_c2 is not None

        # print(f"{loss_a=}")
        # print(f"{loss_c=}")
        # print(f"{grad_a1=}")
        # print(f"{grad_c1=}")
        # see notes for B
        torch_assert_equal(loss_a, loss_c)
        torch_assert_close(grad_a1, grad_c1)  #, rtol=1e-03, atol=1e-04)
        torch_assert_close(grad_a2, grad_c2)  #, rtol=1e-03, atol=1e-04)
