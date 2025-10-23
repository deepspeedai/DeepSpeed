# Copyright (c) The DeepSpeed Contributors
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
"""
Arctic Long Sequence Training (ALST) Tiled compute component tests
"""

from deepspeed.runtime.sequence_parallel.ulysses_sp import TiledMLP, sequence_tiled_compute, TiledFusedLogitsLoss
from deepspeed.utils import safe_get_full_grad
from torch.nn import Linear, Module
from unit.common import DistributedTest, preferred_dtype
from unit.util import torch_assert_equal, torch_assert_close, CaptureStderr
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

    def __init__(self, hidden_dim, vocab_size):
        super().__init__()
        self.vocab_size = vocab_size
        # Critical - need to use a stack of at least 2 mlps to validate that the backward of the last mlp sends the correct gradients to the previous mlp in the stack
        self.mlp1 = SimpleMLP(hidden_dim)
        self.mlp2 = SimpleMLP(hidden_dim)
        self.lm_head = torch.nn.Linear(hidden_dim, vocab_size, bias=False)
        self.cross_entropy_loss = torch.nn.CrossEntropyLoss()

    def forward(self, x, y):
        x = self.mlp1(x)
        x = self.mlp2(x)
        logits = self.lm_head(x)
        return self.cross_entropy_loss(logits.view(-1, self.vocab_size), y.view(-1))


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


@pytest.mark.parametrize("batch_size", [1, 2])
@pytest.mark.parametrize("zero_stage", [2, 3])
class TestTiledCompute(DistributedTest):
    world_size = 1

    def test_tiled_mlp(self, zero_stage, batch_size):

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

        vocab_size = 10
        seed = 42
        hidden_dim = 128
        bs = batch_size
        seqlen = 125  # use a non 2**n length to test varlen shards (last short)
        torch.manual_seed(seed)
        x = torch.rand((bs, seqlen, hidden_dim), dtype=dtype, requires_grad=True)
        y = torch.empty((bs, seqlen), dtype=torch.long, requires_grad=False).random_(vocab_size)

        # A. Baseline: model with normal MLP
        torch.manual_seed(seed)
        model_a = MyModel(hidden_dim=hidden_dim, vocab_size=vocab_size).to(dtype)
        model_a, _, _, _ = deepspeed.initialize(config=config_dict,
                                                model=model_a,
                                                model_parameters=model_a.parameters())

        x = x.to(model_a.device)
        y = y.to(model_a.device)

        x_a = x.clone().detach().requires_grad_(True)
        y_a = y.clone().detach()

        loss_a = model_a(x_a, y_a)
        model_a.backward(loss_a)
        param_grad_a1 = get_grad(model_a.module.mlp1.up_proj.weight, zero_stage)
        param_grad_a2 = get_grad(model_a.module.mlp2.up_proj.weight, zero_stage)
        x_grad_a = x_a.grad
        assert param_grad_a1 is not None
        assert param_grad_a2 is not None
        assert x_grad_a is not None

        # B. model with tiled MLP using TiledMLP
        torch.manual_seed(seed)
        SimpleMLP.forward = mlp_forward_tiled_mlp
        model_b = MyModel(hidden_dim=hidden_dim, vocab_size=vocab_size).to(dtype)
        model_b, _, _, _ = deepspeed.initialize(config=config_dict,
                                                model=model_b,
                                                model_parameters=model_b.parameters())

        x_b = x.clone().detach().requires_grad_(True)
        y_b = y.clone().detach()
        loss_b = model_b(x_b, y_b)

        with CaptureStderr() as cs:
            model_b.backward(loss_b)
        # see the explanation inside TiledMLP.backward
        assert "grad and param do not obey the gradient layout contract" not in cs.err, f"stride issue: {cs.err}"

        param_grad_b1 = get_grad(model_b.module.mlp1.up_proj.weight, zero_stage)
        param_grad_b2 = get_grad(model_b.module.mlp2.up_proj.weight, zero_stage)
        x_grad_b = x_b.grad
        assert param_grad_b1 is not None
        assert param_grad_b2 is not None
        assert x_grad_b is not None

        # print(f"{loss_a=}")
        # print(f"{loss_b=}")
        # print(f"{param_grad_a1=}")
        # print(f"{param_grad_b1=}")
        # print(f"{param_grad_a2=}")
        # print(f"{param_grad_b2=}")
        torch_assert_equal(loss_a, loss_b)

        # Gradient will not be exactly the same, especially under half-precision. And bf16 is
        # particularly lossy so need to lower tolerance a bit more than the default. Switch to
        # dtype torch.float or even torch.double to see that the diff is tiny - so the math is
        # correct, but accumulation error adds up. Alternatively making hidden_dim bigger makes the
        # divergence much smaller as well.
        torch_assert_close(param_grad_a1, param_grad_b1)  #, rtol=1e-03, atol=1e-04)
        torch_assert_close(param_grad_a2, param_grad_b2)  #, rtol=1e-03, atol=1e-04)
        torch_assert_close(x_grad_a, x_grad_b)

        # C. model with tiled MLP using the generic version of the same via sequence_tiled_compute + SequenceTiledCompute
        torch.manual_seed(seed)
        SimpleMLP.forward = mlp_forward_sequence_tiled_compute
        model_c = MyModel(hidden_dim=hidden_dim, vocab_size=vocab_size).to(dtype)
        model_c, _, _, _ = deepspeed.initialize(config=config_dict,
                                                model=model_c,
                                                model_parameters=model_c.parameters())

        x_c = x.clone().detach().requires_grad_(True)
        y_c = y.clone().detach()
        loss_c = model_c(x_c, y_c)
        with CaptureStderr() as cs:
            model_c.backward(loss_c)

        assert "grad and param do not obey the gradient layout contract" not in cs.err, f"stride issue: {cs.err}"

        param_grad_c1 = get_grad(model_c.module.mlp1.up_proj.weight, zero_stage)
        param_grad_c2 = get_grad(model_c.module.mlp2.up_proj.weight, zero_stage)
        x_grad_c = x_c.grad
        assert param_grad_c1 is not None
        assert param_grad_c2 is not None
        assert x_grad_c is not None

        # print(f"{loss_a=}")
        # print(f"{loss_c=}")
        # print(f"{param_grad_a1=}")
        # print(f"{param_grad_c1=}")
        # see notes for B
        torch_assert_equal(loss_a, loss_c)
        torch_assert_close(param_grad_a1, param_grad_c1)  #, rtol=1e-03, atol=1e-04)
        torch_assert_close(param_grad_a2, param_grad_c2)  #, rtol=1e-03, atol=1e-04)
        torch_assert_close(x_grad_a, x_grad_c)


@pytest.mark.parametrize("batch_size", [1, 2])
@pytest.mark.parametrize("zero_stage", [2, 3])
class TestTiledFusedLogitsLoss(DistributedTest):
    world_size = 1

    def test_tiled_fused_logits_loss(self, zero_stage, batch_size):

        def tiled_forward(self, x, y):
            x = self.mlp1(x)
            x = self.mlp2(x)

            def loss_fn(self, x, y):
                logits = self.lm_head(x)
                return self.cross_entropy_loss(logits.view(-1, self.vocab_size), y.view(-1))

            mask = None
            shards = 2
            compute_params = [self.lm_head.weight]
            output_reduction = "mean"
            loss = TiledFusedLogitsLoss.apply(
                loss_fn,
                self,
                x,
                y,
                mask,
                shards,
                compute_params,
                output_reduction,
            )
            return loss

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
        #dtype = torch.float
        if dtype == torch.bfloat16:
            config_dict["bf16"] = {"enabled": True}
        elif dtype == torch.float16:
            config_dict["fp16"] = {"enabled": True, "loss_scale": 1.0}

        # for debug
        # torch.set_printoptions(precision=8, sci_mode=True)

        vocab_size = 100
        seed = 42
        hidden_dim = 64
        bs = batch_size
        seqlen = 425  # use a non 2**n length to test varlen shards (last short)
        torch.manual_seed(seed)
        x = torch.rand((bs, seqlen, hidden_dim), dtype=dtype, requires_grad=True)
        y = torch.empty((bs, seqlen), dtype=torch.long, requires_grad=False).random_(vocab_size)

        # A. Baseline: model with normal loss
        torch.manual_seed(seed)
        model_a = MyModel(hidden_dim=hidden_dim, vocab_size=vocab_size).to(dtype)
        model_a, _, _, _ = deepspeed.initialize(config=config_dict,
                                                model=model_a,
                                                model_parameters=model_a.parameters())

        x = x.to(model_a.device)
        y = y.to(model_a.device)

        x_a = x.clone().detach().requires_grad_(True)
        y_a = y.clone().detach()

        loss_a = model_a(x_a, y_a)
        model_a.backward(loss_a)
        param_grad_a = get_grad(model_a.module.lm_head.weight, zero_stage)
        x_grad_a = x_a.grad
        assert param_grad_a is not None
        assert x_grad_a is not None

        # B. model with fused tiled logits loss
        torch.manual_seed(seed)
        MyModel.forward_orig = MyModel.forward
        MyModel.forward = tiled_forward
        model_b = MyModel(hidden_dim=hidden_dim, vocab_size=vocab_size).to(dtype)
        model_b, _, _, _ = deepspeed.initialize(config=config_dict,
                                                model=model_b,
                                                model_parameters=model_b.parameters())

        x_b = x.clone().detach().requires_grad_(True)
        y_b = y.clone().detach()
        loss_b = model_b(x_b, y_b)

        with CaptureStderr() as cs:
            model_b.backward(loss_b)
        # see the explanation inside TiledMLP.backward
        assert "grad and param do not obey the gradient layout contract" not in cs.err, f"stride issue: {cs.err}"

        param_grad_b = get_grad(model_b.module.lm_head.weight, zero_stage)
        x_grad_b = x_b.grad
        assert param_grad_b is not None
        assert x_grad_b is not None

        # print(f"{loss_a=}")
        # print(f"{loss_b=}")
        # print(f"{x_grad_a=}")
        # print(f"{x_grad_b=}")
        # print(f"{param_grad_a=}")
        # print(f"{param_grad_b=}")
        # usually this is an exact match, but on cpu CI this fails.
        torch_assert_close(loss_a, loss_b)

        # Gradient will not be exactly the same, especially under half-precision. And bf16 is
        # particularly lossy so need to lower tolerance a bit more than the default. Switch to
        # dtype torch.float or even torch.double to see that the diff is tiny - so the math is
        # correct, but accumulation error adds up. Alternatively making hidden_dim bigger makes the
        # divergence much smaller as well.
        torch_assert_close(x_grad_a, x_grad_b)
        torch_assert_close(param_grad_a, param_grad_b)  #, rtol=1e-03, atol=1e-04)

        # restore
        MyModel.forward = MyModel.forward_orig
