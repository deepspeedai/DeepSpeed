# Copyright (c) The DeepSpeed Contributors
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
"""
UlyssesPlus: Tiled compute tests
"""

from deepspeed.runtime.sequence_parallel.ulysses_sp import TiledMLP, sequence_tiled_compute
from deepspeed.utils import safe_get_full_grad
from torch.nn import Linear, Module
from unit.common import DistributedTest, preferred_dtype
import deepspeed
import pytest
import torch


def torch_assert_equal(actual, expected, **kwargs):
    """
    Compare two tensors or non-tensor numbers for their equality.
    Add msg=blah to add an additional comment to when assert fails.
    """
    return torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0, **kwargs)


def torch_assert_close(actual, expected, **kwargs):
    """
    Compare two tensors or non-tensor numbers for their closeness.

    Add msg=blah to add an additional comment to when assert fails.

    For default values of `rtol` and `atol` which are dtype dependent, see the table at https://docs.pytorch.org/docs/stable/testing.html#torch.testing.assert_close
    For example for bf16 it is `rtol=1.6e-2` and `atol=1e-5`.

    The check doesn't assert when `|a - b| <= (atol + rtol * |b|)`
    """
    return torch.testing.assert_close(actual, expected, **kwargs)


def get_grad(param, zero_stage):
    if zero_stage == 1:
        return param.grad
    else:
        return safe_get_full_grad(param)


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
        self.mlp = SimpleMLP(hidden_dim)
        self.cross_entropy_loss = torch.nn.CrossEntropyLoss()

    def forward(self, x, y):
        x = self.mlp(x)
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
        if dtype == torch.float16:
            config_dict["fp16"] = {"enabled": True, "loss_scale": 1.0}
        elif dtype == torch.bfloat16:
            config_dict["bf16"] = {"enabled": True}

        #dtype = torch.float
        torch.set_printoptions(precision=8, sci_mode=True)

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
        grad_a = get_grad(model_a.module.mlp.up_proj.weight, zero_stage)
        assert grad_a is not None

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
        grad_b = get_grad(model_b.module.mlp.up_proj.weight, zero_stage)
        assert grad_b is not None

        print(f"{loss_a=}")
        print(f"{loss_b=}")
        print(f"{grad_a=}")
        print(f"{grad_b=}")
        torch_assert_equal(loss_a, loss_b)

        # Gradient will not be exactly the same, especially under half-precision. And bf16 is
        # particularly lossy so need to lower tolerance a bit more than the default. Switch to
        # dtype torch.float or even torch.double to see that the diff is tiny - so the math is
        # correct, but accumulation error adds up. Alternatively making hidden_dim bigger makes the
        # divergence much smaller as well.
        torch_assert_close(grad_a, grad_b, rtol=1e-03, atol=1e-04)

        #die

        # C. model with tiled MLP using sequence_tiled_compute + SequenceTiledCompute
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
        grad_c = get_grad(model_c.module.mlp.up_proj.weight, zero_stage)
        assert grad_c is not None

        print(f"{loss_a=}")
        print(f"{loss_c=}")
        print(f"{grad_a=}")
        print(f"{grad_c=}")
        # see notes for B
        torch_assert_equal(loss_a, loss_c)
        torch_assert_close(grad_a, grad_c, rtol=1e-03, atol=1e-04)

        #die
