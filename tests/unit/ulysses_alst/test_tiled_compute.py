# Copyright (c) The DeepSpeed Contributors
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
"""
Arctic Long Sequence Training (ALST) Tiled compute component tests
"""

from deepspeed.runtime.activation_checkpointing.tiled_rematerialization import (
    tiled_rematerialize, tiled_rematerialize_compute, classify_rematerialize_kwarg, _hold_gathered_params,
    plan_packed_tiles, rebase_cu_seqlens, classify_packed_kwarg, packed_seq_tiled_rematerialize_compute,
    PackedSeqTiledRematerializeCompute, plan_chain_tiles, _merge_smallest_chain_pair, _causal_conv_carry,
    sequential_chain_rematerialize, qwen35_gdn_tile_forward, limit_grad_reduce)
from deepspeed.runtime.sequence_parallel.ulysses_sp import TiledMLP, sequence_tiled_compute, TiledFusedLogitsLoss
from deepspeed.accelerator import get_accelerator
from deepspeed.utils import safe_get_full_grad
from torch.nn import Linear, Module
from unit.common import DistributedTest, preferred_dtype
from unit.util import torch_assert_equal, torch_assert_close, CaptureStderr
import contextlib
import functools
import importlib.util
import inspect
import os
import sys
import deepspeed
import deepspeed.comm as dist
import pytest
import torch


def test_nested_grad_reduce_waits_for_outer_shard():
    # Inner last shard must not look ready while an outer batch shard remains.
    weight = torch.nn.Parameter(torch.zeros(1))
    with limit_grad_reduce(False):
        with limit_grad_reduce(True) as ready:
            weight.ds_grad_is_ready = ready
    assert weight.ds_grad_is_ready is False
    with limit_grad_reduce(True):
        with limit_grad_reduce(True) as ready:
            weight.ds_grad_is_ready = ready
    assert weight.ds_grad_is_ready is True


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
@pytest.mark.parametrize("seqlen,shards", [(5, 4), (2, 4), (7, 4), (8, 4)])
@pytest.mark.parametrize("output_reduction", [None, "sum", "mean"])
def test_sequence_tiled_compute_shard_boundaries(batch_size, seqlen, shards, output_reduction):
    # Empty trailing tiles must preserve both gradients and updates relative to untiled computation.
    device = get_accelerator().device_name()
    dtype = torch.float32
    hidden_dim = 8
    torch.manual_seed(42)
    tiled_model = SimpleMLP(hidden_dim).to(device=device, dtype=dtype)
    reference_model = SimpleMLP(hidden_dim).to(device=device, dtype=dtype)
    reference_model.load_state_dict(tiled_model.state_dict())
    tiled_optimizer = torch.optim.SGD(tiled_model.parameters(), lr=0.01)
    reference_optimizer = torch.optim.SGD(reference_model.parameters(), lr=0.01)

    def compute(x, scale, model):
        output = mlp_forward_orig(model, x) * scale
        return output if output_reduction is None else output.sum()

    for _ in range(2):
        tiled_optimizer.zero_grad()
        reference_optimizer.zero_grad()
        x = torch.rand((batch_size, seqlen, hidden_dim), device=device, dtype=dtype, requires_grad=True)
        x_reference = x.detach().clone().requires_grad_(True)
        scale = torch.rand_like(x)

        output = sequence_tiled_compute(
            compute,
            seqlen,
            shards,
            kwargs_to_shard={
                "x": x,
                "scale": scale
            },
            kwargs_to_pass={"model": tiled_model},
            grad_requiring_tensor_key="x",
            compute_params=list(tiled_model.parameters()),
            output_unshard_dimension=1 if output_reduction is None else 0,
            output_reduction=output_reduction,
        )
        expected = mlp_forward_orig(reference_model, x_reference) * scale
        if output_reduction is not None:
            expected = expected.sum()
            if output_reduction == "mean":
                # Averaging per-tile sums divides the full sum by the requested tile count.
                expected = expected / shards

        torch.testing.assert_close(output, expected)
        output.sum().backward()
        expected.sum().backward()
        torch.testing.assert_close(x.grad, x_reference.grad)
        for param, reference_param in zip(tiled_model.parameters(), reference_model.parameters()):
            torch.testing.assert_close(param.grad, reference_param.grad)

        tiled_optimizer.step()
        reference_optimizer.step()
        for param, reference_param in zip(tiled_model.parameters(), reference_model.parameters()):
            torch.testing.assert_close(param, reference_param)


@pytest.mark.parametrize("shards", [2, 4])
@pytest.mark.parametrize("layout", ["transposed", "channel_slice"])
class TestTiledMLPInputLayout:
    """
    A caller may hand TiledMLP an activation that is not contiguous: a transposed tensor, or a slice along the
    hidden dimension of a wider one. The backward flattens batch and sequence into one axis, and for these
    layouts that flattening needs a copy, so it must not be expressed as a view.
    """

    def make_input(self, layout, batch_size, seqlen, hidden_dim, dtype):
        if layout == "transposed":
            # [bs, hidden, seqlen] transposed into [bs, seqlen, hidden] keeps the original strides
            source = torch.rand((batch_size, hidden_dim, seqlen), dtype=dtype)
            x = source.transpose(1, 2)
        else:
            # a hidden-dimension slice of a wider activation keeps the wider row stride
            source = torch.rand((batch_size, seqlen, hidden_dim * 2), dtype=dtype)
            x = source[..., :hidden_dim]
        assert not x.is_contiguous(), f"{layout} input is contiguous, so it does not exercise the flattening"
        # detach preserves the strides and makes this a leaf, so x.grad is populated with the same layout
        return x.detach().requires_grad_(True)

    def test_tiled_mlp_accepts_a_non_contiguous_input(self, layout, shards, batch_size=2, seqlen=12, hidden_dim=16):
        dtype = torch.float32
        torch.manual_seed(42)
        mlp = SimpleMLP(hidden_dim).to(dtype)
        compute_params = [mlp.down_proj.weight, mlp.up_proj.weight]

        x_tiled = self.make_input(layout, batch_size, seqlen, hidden_dim, dtype)
        x_reference = x_tiled.clone().detach().requires_grad_(True)

        output_tiled = TiledMLP.apply(mlp_forward_orig, mlp, x_tiled, shards, compute_params)
        output_tiled.sum().backward()
        grads_tiled = [p.grad.clone() for p in compute_params]
        for p in compute_params:
            p.grad = None

        output_reference = mlp_forward_orig(mlp, x_reference)
        output_reference.sum().backward()

        torch_assert_close(output_tiled, output_reference)
        torch_assert_close(x_tiled.grad, x_reference.grad)
        for grad_tiled, param in zip(grads_tiled, compute_params):
            torch_assert_close(grad_tiled, param.grad)


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


@pytest.mark.parametrize("shards", [2, 4])
class TestTiledFusedLogitsLossInputLayout:
    """
    Same caller contract as TestTiledMLPInputLayout, on the loss. TiledFusedLogitsLoss flattens batch and
    sequence into one axis before sharding, and a transposed activation cannot be flattened by a view.
    """

    def make_model(self, hidden_dim, vocab_size, dtype):
        model = Linear(hidden_dim, vocab_size, bias=False, dtype=dtype)
        torch.nn.init.normal_(model.weight, std=0.02)
        return model

    @staticmethod
    def loss_fn(model, x_shard, y_shard):
        return torch.nn.functional.cross_entropy(model(x_shard), y_shard, reduction="sum")

    def test_transposed_input_matches_a_contiguous_copy(self,
                                                        shards,
                                                        batch_size=2,
                                                        seqlen=12,
                                                        hidden_dim=16,
                                                        vocab_size=32):
        dtype = torch.float32
        torch.manual_seed(0)
        # [bs, hidden, seqlen] transposed into [bs, seqlen, hidden] keeps the original strides
        source = torch.rand((batch_size, hidden_dim, seqlen), dtype=dtype)
        strided = source.transpose(1, 2).detach().requires_grad_(True)
        assert not strided.is_contiguous(), "input is contiguous, so it does not exercise the flattening"
        contiguous = strided.detach().clone().contiguous().requires_grad_(True)
        y = torch.randint(0, vocab_size, (batch_size, seqlen))

        model = self.make_model(hidden_dim, vocab_size, dtype)
        losses, param_grads = [], []
        for x in (strided, contiguous):
            model.zero_grad()
            loss = TiledFusedLogitsLoss.apply(self.loss_fn, model, x, y, None, shards, list(model.parameters()), "sum")
            # The backward scatters into a zeros_like of the same flattened activation, so the
            # gradient path runs through the flatten under test as well as the forward.
            loss.backward()
            losses.append(loss)
            param_grads.append([p.grad.detach().clone() for p in model.parameters()])

        torch_assert_close(losses[0], losses[1])
        torch_assert_close(strided.grad, contiguous.grad)
        for grad_a, grad_b in zip(*param_grads):
            torch_assert_close(grad_a, grad_b)


class SeqMix(Module):
    """Sequence-mixing and batch-independent: cumsum over S, then a linear."""

    def __init__(self, hidden_dim):
        super().__init__()
        self.proj = Linear(hidden_dim, hidden_dim, bias=False)

    def forward(self, x, mask=None, bias=None, position_embeddings=None):
        y = x.cumsum(dim=1)
        if mask is not None:
            y = y * mask.unsqueeze(-1)
        if bias is not None:
            y = y + bias
        if position_embeddings is not None:
            # Batch-major RoPE must be chunked with hidden; a leftover B=2
            # tuple on a B=1 row fails here the same way Qwen3.5 does.
            cos, sin = position_embeddings
            y = y * cos + sin
        return self.proj(y)


seqmix_forward_orig = SeqMix.forward


class StackedSeqMix(Module):

    def __init__(self, hidden_dim):
        super().__init__()
        self.mix1 = SeqMix(hidden_dim)
        self.mix2 = SeqMix(hidden_dim)

    def forward(self, x):
        x = self.mix1(x)
        x = self.mix2(x)
        return x.square().mean()


def _zero_config(zero_stage, micro_batch=1, dtype=None):
    config_dict = {
        "train_micro_batch_size_per_gpu": micro_batch,
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
    if dtype is None:
        dtype = preferred_dtype()
    if dtype == torch.bfloat16:
        config_dict["bf16"] = {"enabled": True}
    elif dtype == torch.float16:
        config_dict["fp16"] = {"enabled": True, "loss_scale": 1.0}
    return config_dict, dtype


def _assert_compute_params_ready(params):
    for param in params:
        assert getattr(param, "ds_grad_is_ready", True) is True, param


def test_classify_rematerialize_kwarg():
    batch = 3
    hidden = torch.randn(batch, 4, 8)
    assert classify_rematerialize_kwarg("mask", torch.ones(batch, 4), batch) == "shard"
    assert classify_rematerialize_kwarg("bias", torch.ones(1, 1, 8), batch) == "pass"
    assert classify_rematerialize_kwarg("scale", 1.0, batch) == "pass"
    assert classify_rematerialize_kwarg("flag", None, batch) == "pass"
    assert classify_rematerialize_kwarg("past_key_values", None, batch) == "pass"
    pos = (torch.randn(1, 4, 8), torch.randn(1, 4, 8))
    assert classify_rematerialize_kwarg("position_embeddings", pos, batch) == "pass"
    pos_b = (torch.randn(batch, 4, 8), torch.randn(batch, 4, 8))
    assert classify_rematerialize_kwarg("position_embeddings", pos_b, batch) == "shard"
    with pytest.raises(TypeError, match="cu_seqlens"):
        classify_rematerialize_kwarg("cu_seqlens", torch.arange(batch + 1), batch)
    with pytest.raises(TypeError, match="cache"):
        classify_rematerialize_kwarg("past_key_value", object(), batch)
    with pytest.raises(TypeError, match="neither batch"):
        classify_rematerialize_kwarg("odd", torch.ones(2, 4), batch)
    assert hidden.shape[0] == batch


def test_plan_packed_tiles():
    # (a) single sample: one tile spanning the whole row, regardless of budget.
    cu_seqlens = torch.tensor([0, 5])
    assert plan_packed_tiles(cu_seqlens, token_budget=1) == [(0, 1, 0, 5)]

    # (b) even samples under budget: greedy grouping, split only when exceeding.
    cu_seqlens = torch.tensor([0, 3, 6, 9])
    assert plan_packed_tiles(cu_seqlens, token_budget=7) == [(0, 2, 0, 6), (2, 3, 6, 9)]

    # (c) one sample exceeds the budget alone: its own oversized tile, never split.
    cu_seqlens = torch.tensor([0, 2, 12, 14])
    assert plan_packed_tiles(cu_seqlens, token_budget=5) == [(0, 1, 0, 2), (1, 2, 2, 12), (2, 3, 12, 14)]

    # (d) shards derives the same boundaries as the equivalent token_budget.
    cu_seqlens = torch.tensor([0, 4, 8, 12, 16])
    by_shards = plan_packed_tiles(cu_seqlens, shards=2)
    by_budget = plan_packed_tiles(cu_seqlens, token_budget=8)
    assert by_shards == by_budget == [(0, 2, 0, 8), (2, 4, 8, 16)]

    # shards upper-bounds the tile count even when token_budget would split further.
    cu_seqlens = torch.tensor([0, 3, 6, 9, 12])
    assert plan_packed_tiles(cu_seqlens, shards=2, token_budget=3) == [(0, 1, 0, 3), (1, 4, 3, 12)]

    with pytest.raises(ValueError, match="shards or token_budget"):
        plan_packed_tiles(torch.tensor([0, 3, 6, 9]))


def test_rebase_cu_seqlens():
    cu_seqlens = torch.tensor([0, 4, 8, 12, 16])
    rebased = rebase_cu_seqlens(cu_seqlens, 1, 3)
    torch_assert_equal(rebased, torch.tensor([0, 4, 8]))
    assert int(rebased[-1]) == int(cu_seqlens[3] - cu_seqlens[1])


def test_classify_packed_kwarg():
    num_samples = 3
    assert classify_packed_kwarg("cu_seqlens", torch.arange(num_samples + 1), num_samples) == "pack_meta"
    assert classify_packed_kwarg("cu_seq_lens_q", torch.arange(num_samples + 1), num_samples) == "pack_meta"
    assert classify_packed_kwarg("max_seqlen_q", 8, num_samples) == "pack_meta"
    assert classify_packed_kwarg("seq_idx", torch.zeros(12, dtype=torch.long), num_samples) == "pack_meta"
    with pytest.raises(TypeError, match="expected num_samples\\+1"):
        classify_packed_kwarg("cu_seqlens", torch.arange(num_samples), num_samples)

    total_tokens = 12
    assert classify_packed_kwarg("position_ids", torch.arange(total_tokens), num_samples) == "token"
    assert classify_packed_kwarg("attention_mask", torch.ones(total_tokens, 4), num_samples) == "token"
    pos = (torch.randn(total_tokens, 8), torch.randn(total_tokens, 8))
    assert classify_packed_kwarg("position_embeddings", pos, num_samples) == "token"

    assert classify_packed_kwarg("bias", torch.ones(1, 1, 8), num_samples) == "pass"
    assert classify_packed_kwarg("scale", 1.0, num_samples) == "pass"
    assert classify_packed_kwarg("flag", None, num_samples) == "pass"
    with pytest.raises(TypeError, match="cache"):
        classify_packed_kwarg("past_key_value", object(), num_samples)

    # The batch-axis classifier is untouched: it still hard-rejects cu_seqlens.
    with pytest.raises(TypeError, match="cu_seqlens"):
        classify_rematerialize_kwarg("cu_seqlens", torch.arange(num_samples + 1), num_samples)


class PackedSeqMix(Module):
    """Packed-sequence mixer: cumsum per sample, reset at cu_seqlens boundaries.

    A wrong (unrebased or mis-sliced) cu_seqlens misaligns the per-sample
    slice against the tile's own x, so this catches rebasing bugs directly
    rather than only via an end-to-end numerical parity check.
    """

    def __init__(self, hidden_dim):
        super().__init__()
        self.proj = Linear(hidden_dim, hidden_dim, bias=False)

    def forward(self, x, cu_seqlens, position_ids=None):
        parts = []
        for i in range(cu_seqlens.shape[0] - 1):
            start_tok, end_tok = int(cu_seqlens[i]), int(cu_seqlens[i + 1])
            parts.append(x[start_tok:end_tok].cumsum(dim=0))
        y = torch.cat(parts, dim=0)
        if position_ids is not None:
            y = y + position_ids.unsqueeze(-1).to(y.dtype)
        return self.proj(y)


packedseqmix_forward_orig = PackedSeqMix.forward


@pytest.mark.parametrize("sample_lens,token_budget", [
    ((4, 4, 4), 8),
    ((6, ), 3),
    ((2, 10, 2), 5),
])
@pytest.mark.parametrize("zero_stage", [2, 3])
class TestPackedSeqTiledRematerialize(DistributedTest):
    world_size = 1

    def test_packed_seq_tiled_rematerialize_parity(self, zero_stage, sample_lens, token_budget):
        config_dict, dtype = _zero_config(zero_stage, micro_batch=1, dtype=torch.float32)
        seed = 11
        hidden_dim = 16
        total_tokens = sum(sample_lens)
        cu_seqlens = torch.tensor([0] + list(torch.tensor(sample_lens).cumsum(0)), dtype=torch.long)
        torch.manual_seed(seed)
        x = torch.rand((total_tokens, hidden_dim), dtype=dtype, requires_grad=True)
        position_ids = torch.cat([torch.arange(length, dtype=dtype)
                                  for length in sample_lens]).unsqueeze(-1).expand(-1, hidden_dim)[:, 0]

        torch.manual_seed(seed)
        model_a = PackedSeqMix(hidden_dim).to(dtype)
        model_a, _, _, _ = deepspeed.initialize(config=config_dict,
                                                model=model_a,
                                                model_parameters=model_a.parameters())
        x = x.to(model_a.device)
        cu_seqlens = cu_seqlens.to(model_a.device)
        position_ids = position_ids.to(model_a.device)
        x_a = x.clone().detach().requires_grad_(True)
        loss_a = model_a(x_a, cu_seqlens, position_ids=position_ids).square().mean()
        model_a.backward(loss_a)
        grad_a = get_grad(model_a.module.proj.weight, zero_stage)

        torch.manual_seed(seed)
        model_b = PackedSeqMix(hidden_dim).to(dtype)
        model_b, _, _, _ = deepspeed.initialize(config=config_dict,
                                                model=model_b,
                                                model_parameters=model_b.parameters())

        def model_forward(self, hidden, seqlens, position_ids=None):
            return packed_seq_tiled_rematerialize_compute(
                lambda h, cu_seqlens=None, position_ids=None: packedseqmix_forward_orig(
                    self, h, cu_seqlens, position_ids=position_ids),
                hidden,
                seqlens,
                token_budget=token_budget,
                compute_params=[self.proj.weight],
                kwargs_to_token_shard=dict(position_ids=position_ids),
            )

        model_b.module.forward = model_forward.__get__(model_b.module, PackedSeqMix)
        x_b = x.clone().detach().requires_grad_(True)
        loss_b = model_b(x_b, cu_seqlens, position_ids=position_ids).square().mean()
        with CaptureStderr() as cs:
            model_b.backward(loss_b)
        assert "grad and param do not obey the gradient layout contract" not in cs.err, cs.err
        _assert_compute_params_ready([model_b.module.proj.weight])
        torch_assert_close(loss_a, loss_b)
        torch_assert_close(grad_a, get_grad(model_b.module.proj.weight, zero_stage))
        torch_assert_close(x_a.grad, x_b.grad)


class TestPackedSeqTiledRematerializeNoOp(DistributedTest):
    world_size = 1

    def test_single_sample_is_bitwise_noop(self):
        hidden_dim = 8
        seqlen = 5
        cu_seqlens = torch.tensor([0, seqlen], dtype=torch.long)
        x = torch.rand((seqlen, hidden_dim))
        calls = []
        orig_apply = PackedSeqTiledRematerializeCompute.apply

        def spy_apply(*args, **kwargs):
            calls.append(1)
            return orig_apply(*args, **kwargs)

        PackedSeqTiledRematerializeCompute.apply = staticmethod(spy_apply)
        try:
            direct = x + cu_seqlens[-1].to(x.dtype)
            tiled = packed_seq_tiled_rematerialize_compute(lambda h, cu_seqlens=None: h + cu_seqlens[-1].to(h.dtype),
                                                           x,
                                                           cu_seqlens,
                                                           token_budget=1)
        finally:
            PackedSeqTiledRematerializeCompute.apply = orig_apply
        torch_assert_equal(direct, tiled)
        assert not calls, "single-sample cu_seqlens must not invoke PackedSeqTiledRematerializeCompute.apply"


@pytest.mark.parametrize("zero_stage", [2, 3])
class TestPackedSeqTiledRematerializeDistributedAgree(DistributedTest):
    world_size = 2

    def test_agreed_tile_count_matches_across_ranks(self, zero_stage):
        from deepspeed.runtime.activation_checkpointing import tiled_rematerialization as tiled_remat_mod

        config_dict, dtype = _zero_config(zero_stage, micro_batch=1, dtype=torch.float32)
        hidden_dim = 16
        rank = dist.get_rank()
        # Different sample-length histograms per rank -> different local tile
        # counts under the same token budget, forcing a real merge on
        # whichever rank's plan is finer than the cross-rank minimum.
        sample_lens = (2, 2, 2, 2) if rank == 0 else (3, 3, 2)
        total_tokens = sum(sample_lens)
        cu_seqlens = torch.tensor([0] + list(torch.tensor(sample_lens).cumsum(0)), dtype=torch.long)

        torch.manual_seed(11)
        model = PackedSeqMix(hidden_dim).to(dtype)
        model, _, _, _ = deepspeed.initialize(config=config_dict, model=model, model_parameters=model.parameters())
        cu_seqlens = cu_seqlens.to(model.device)

        def model_forward(self, hidden, seqlens, token_budget):
            return packed_seq_tiled_rematerialize_compute(
                lambda h, cu_seqlens=None: packedseqmix_forward_orig(self, h, cu_seqlens),
                hidden,
                seqlens,
                token_budget=token_budget,
                compute_params=[self.proj.weight],
            )

        model.module.forward = model_forward.__get__(model.module, PackedSeqMix)

        agreed_counts = []
        orig_agree = tiled_remat_mod._agree_packed_tile_count

        def spy_agree(tiles, device):
            merged = orig_agree(tiles, device)
            agreed_counts.append(len(merged))
            return merged

        tiled_remat_mod._agree_packed_tile_count = spy_agree
        try:
            x1 = torch.rand((total_tokens, hidden_dim), dtype=dtype, device=model.device, requires_grad=True)
            loss1 = model(x1, cu_seqlens, token_budget=3).square().mean()
            model.backward(loss1)
            _assert_compute_params_ready([model.module.proj.weight])
            model.step()

            # A different local plan on the same rank (still local_count 4 vs
            # 3 for token_budget=3, but 2 vs 3 here) must not reuse the first
            # call's agreed count -- this path is deliberately uncached.
            x2 = torch.rand((total_tokens, hidden_dim), dtype=dtype, device=model.device, requires_grad=True)
            loss2 = model(x2, cu_seqlens, token_budget=4).square().mean()
            model.backward(loss2)
            _assert_compute_params_ready([model.module.proj.weight])
        finally:
            tiled_remat_mod._agree_packed_tile_count = orig_agree

        # token_budget=3: local counts are 4 (rank0) vs 3 (rank1) -> min=3.
        # token_budget=4: local counts are 2 (rank0) vs 3 (rank1) -> min=2.
        assert agreed_counts == [3, 2], agreed_counts

    def test_single_sample_rank_joins_agreement(self, zero_stage):
        # A one-sample rank used to return before the MIN all_reduce while its
        # multi-sample peer waited in it, hanging both.
        from deepspeed.runtime.activation_checkpointing import tiled_rematerialization as tiled_remat_mod

        config_dict, dtype = _zero_config(zero_stage, micro_batch=1, dtype=torch.float32)
        hidden_dim = 16
        rank = dist.get_rank()
        sample_lens = (6, ) if rank == 0 else (2, 2, 2)
        cu_seqlens = torch.tensor([0] + list(torch.tensor(sample_lens).cumsum(0)), dtype=torch.long)

        torch.manual_seed(11)
        model = PackedSeqMix(hidden_dim).to(dtype)
        model, _, _, _ = deepspeed.initialize(config=config_dict, model=model, model_parameters=model.parameters())
        cu_seqlens = cu_seqlens.to(model.device)
        x = torch.rand((sum(sample_lens), hidden_dim), dtype=dtype, device=model.device, requires_grad=True)

        def model_forward(self, hidden, seqlens):
            return packed_seq_tiled_rematerialize_compute(
                lambda h, cu_seqlens=None: packedseqmix_forward_orig(self, h, cu_seqlens),
                hidden,
                seqlens,
                token_budget=2,
                compute_params=[self.proj.weight],
            )

        model.module.forward = model_forward.__get__(model.module, PackedSeqMix)

        agreed_counts = []
        apply_calls = []
        orig_agree = tiled_remat_mod._agree_packed_tile_count
        orig_apply = PackedSeqTiledRematerializeCompute.apply

        def spy_agree(tiles, device):
            merged = orig_agree(tiles, device)
            agreed_counts.append(len(merged))
            return merged

        def spy_apply(*args, **kwargs):
            apply_calls.append(1)
            return orig_apply(*args, **kwargs)

        tiled_remat_mod._agree_packed_tile_count = spy_agree
        PackedSeqTiledRematerializeCompute.apply = staticmethod(spy_apply)
        try:
            loss = model(x, cu_seqlens).square().mean()
            model.backward(loss)
        finally:
            tiled_remat_mod._agree_packed_tile_count = orig_agree
            PackedSeqTiledRematerializeCompute.apply = orig_apply

        # Rank 0 cannot split its one sample, so its vote of 1 is the agreed
        # count and rank 1 merges its three samples into one tile.
        assert agreed_counts == [1], agreed_counts
        # The single-sample rank still takes the direct call after agreeing.
        assert len(apply_calls) == (0 if rank == 0 else 1), apply_calls
        _assert_compute_params_ready([model.module.proj.weight])
        assert x.grad is not None


def _hf_style_kernel_fallback(implementation):
    """Mimic transformers' ``use_kernel_func_from_hub_with_fallback`` kwarg filter.

    It keeps only kwargs *named* in the implementation's signature, so an
    implementation that does not name ``cu_seqlens`` / ``seq_idx`` silently
    loses the packed-boundary information.
    """
    applicable_params = tuple(inspect.signature(implementation).parameters)

    @functools.wraps(implementation)
    def wrapped(*args, **kwargs):
        kwargs = {k: v for k, v in kwargs.items() if k in applicable_params}
        return implementation(*args, **kwargs)

    return wrapped


def _decay_scan(x, decay, cu_seqlens=None):
    """fla-like recurrent scan ``h_t = decay_t * h_{t-1} + x_t``, restarted at every cu_seqlens boundary."""
    bounds = [0, x.shape[1]] if cu_seqlens is None else cu_seqlens.tolist()
    outputs = []
    for start_tok, end_tok in zip(bounds[:-1], bounds[1:]):
        state = torch.zeros_like(x[:, 0])
        for t in range(start_tok, end_tok):
            state = decay[:, t] * state + x[:, t]
            outputs.append(state)
    return torch.stack(outputs, dim=1)


def _torch_decay_scan(x, decay, chunk_size=64, **kwargs):
    """Torch-fallback-like scan: names no cu_seqlens, so the state runs across packed samples."""
    return _decay_scan(x, decay)


def _causal_conv(x, weight, seq_idx=None, activation=None):
    """causal_conv1d-like depthwise conv over ``[B, D, T]``; a tap never reaches across a seq_idx change."""
    seqlen = x.shape[-1]
    kernel_size = weight.shape[-1]
    out = torch.zeros_like(x)
    for shift in range(kernel_size):
        shifted = torch.nn.functional.pad(x, (shift, 0))[..., :seqlen]
        if seq_idx is not None:
            shifted_idx = torch.nn.functional.pad(seq_idx, (shift, 0), value=-1)[..., :seqlen]
            same_sample = (shifted_idx == seq_idx).unsqueeze(1).to(x.dtype)
            shifted = shifted * same_sample
        out = out + weight[:, kernel_size - 1 - shift, None] * shifted
    if activation == "silu":
        out = torch.nn.functional.silu(out)
    return out


def _torch_causal_conv(x, weight, activation=None, **kwargs):
    """Torch-fallback-like conv: names no seq_idx, so it slides across packed samples."""
    return _causal_conv(x, weight, activation=activation)


# Same global names as transformers' Qwen3.5 module: PackedGDNMix looks them up
# at call time, and the installer's kernel-path guard inspects them.
torch_chunk_gated_delta_rule = _hf_style_kernel_fallback(_decay_scan)
causal_conv1d_fn = _hf_style_kernel_fallback(_causal_conv)


class PackedGDNMix(Module):
    """Gated-DeltaNet-like mixer with Qwen3_5GatedDeltaNet's call contract.

    ``forward(hidden_states, cache_params, attention_mask, **kwargs)`` runs a
    short causal conv then a decaying recurrent scan, both of which learn the
    packed boundaries only through kwargs (``seq_idx`` / ``cu_seq_lens_q``),
    and returns a bare tensor rather than attention's tuple.
    """

    def __init__(self, hidden_dim, conv_kernel_size=3):
        super().__init__()
        self.in_proj = Linear(hidden_dim, 2 * hidden_dim, bias=False)
        self.conv_weight = torch.nn.Parameter(torch.randn(hidden_dim, conv_kernel_size) * 0.5)
        self.out_proj = Linear(hidden_dim, hidden_dim, bias=False)

    def forward(self, hidden_states, cache_params=None, attention_mask=None, **kwargs):
        assert cache_params is None
        if attention_mask is not None:
            hidden_states = hidden_states * attention_mask.unsqueeze(-1)
        x, gate = self.in_proj(hidden_states).chunk(2, dim=-1)
        x = causal_conv1d_fn(x.transpose(1, 2), self.conv_weight, activation="silu", **kwargs).transpose(1, 2)
        decay = torch.sigmoid(gate)
        y = torch_chunk_gated_delta_rule(x, decay, cu_seqlens=kwargs.pop("cu_seq_lens_q", None), **kwargs)
        return self.out_proj(y)


class PackedAttnMix(Module):
    """Attention-like mixer: per-sample cumsum over ``cu_seq_lens_q``, returns ``(out, None)`` like HF attention."""

    def __init__(self, hidden_dim):
        super().__init__()
        self.proj = Linear(hidden_dim, hidden_dim, bias=False)

    def forward(self, hidden_states, attention_mask=None, past_key_values=None, **kwargs):
        cu_seqlens = kwargs["cu_seq_lens_q"].tolist()
        parts = []
        for start_tok, end_tok in zip(cu_seqlens[:-1], cu_seqlens[1:]):
            parts.append(hidden_states[:, start_tok:end_tok].cumsum(dim=1))
        return self.proj(torch.cat(parts, dim=1)), None


class PackedHybridDecoderLayer(Module):
    """Mirrors Qwen3_5DecoderLayer's mixer call sites, including how each mixer's return is unpacked."""

    def __init__(self, hidden_dim, layer_type):
        super().__init__()
        self.layer_type = layer_type
        if layer_type == "linear_attention":
            self.linear_attn = PackedGDNMix(hidden_dim)
        else:
            self.self_attn = PackedAttnMix(hidden_dim)
        self.mlp = Linear(hidden_dim, hidden_dim, bias=False)

    def forward(self, hidden_states, attention_mask=None, past_key_values=None, **kwargs):
        residual = hidden_states
        if self.layer_type == "linear_attention":
            hidden_states = self.linear_attn(hidden_states=hidden_states,
                                             cache_params=past_key_values,
                                             attention_mask=attention_mask,
                                             **kwargs)
        else:
            hidden_states, _ = self.self_attn(hidden_states=hidden_states,
                                              attention_mask=attention_mask,
                                              past_key_values=past_key_values,
                                              **kwargs)
        hidden_states = residual + hidden_states
        return hidden_states + self.mlp(hidden_states)


class PackedHybridModel(Module):

    def __init__(self, hidden_dim):
        super().__init__()
        layer_types = ("linear_attention", "full_attention", "linear_attention")
        self.layers = torch.nn.ModuleList([PackedHybridDecoderLayer(hidden_dim, t) for t in layer_types])

    def forward(self, hidden_states, **packed_kwargs):
        for layer in self.layers:
            hidden_states = layer(hidden_states, **packed_kwargs)
        return hidden_states


def _packed_hf_kwargs(sample_lens, device):
    cu_seqlens = torch.tensor([0] + list(torch.tensor(sample_lens).cumsum(0)), dtype=torch.int32, device=device)
    seq_idx = torch.cat([torch.full((n, ), i, dtype=torch.int32) for i, n in enumerate(sample_lens)])
    return dict(cu_seq_lens_q=cu_seqlens,
                cu_seq_lens_k=cu_seqlens,
                max_length_q=max(sample_lens),
                max_length_k=max(sample_lens),
                seq_idx=seq_idx.unsqueeze(0).to(device))


def _bench_module():
    # The HF mixer installer lives in a benchmark script, not a package.
    name = "bench_qwen3_offload"
    if name in sys.modules:
        return sys.modules[name]
    repo_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..")
    path = os.path.join(repo_root, "autorun", "bench_qwen3_offload.py")
    if not os.path.exists(path):
        pytest.skip(f"{path} not present")
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    sys.modules[name] = module
    return module


@contextlib.contextmanager
def _packed_mixers_installed(model, token_budget):
    # The installer patches mixer classes, not instances; undo it for later tests.
    saved_forwards = {cls: cls.forward for cls in (PackedGDNMix, PackedAttnMix)}
    try:
        yield _bench_module()._install_packed_seq_tiled_rematerialize(model, token_budget=token_budget)
    finally:
        for cls, forward in saved_forwards.items():
            cls.forward = forward


def test_packed_gdn_mix_baseline_resets_at_boundaries(monkeypatch):
    # Oracle behind the GDN parity tests: with boundary-aware kernels the
    # packed baseline equals running each sample alone; with torch-fallback-like
    # kernels it does not, which is the path the installer must refuse.
    hidden_dim = 8
    sample_lens = (3, 5, 4)
    torch.manual_seed(0)
    model = PackedGDNMix(hidden_dim)
    x = torch.randn(1, sum(sample_lens), hidden_dim)
    packed_kwargs = _packed_hf_kwargs(sample_lens, x.device)

    def max_diff_vs_alone():
        with torch.no_grad():
            packed = model(x, **packed_kwargs)
            alone = torch.cat([model(sample) for sample in x.split(sample_lens, dim=1)], dim=1)
        return (packed - alone).abs().max().item()

    assert max_diff_vs_alone() < 1e-6
    this_module = sys.modules[__name__]
    monkeypatch.setattr(this_module, "torch_chunk_gated_delta_rule", _hf_style_kernel_fallback(_torch_decay_scan))
    assert max_diff_vs_alone() > 1e-3
    monkeypatch.setattr(this_module, "torch_chunk_gated_delta_rule", _hf_style_kernel_fallback(_decay_scan))
    monkeypatch.setattr(this_module, "causal_conv1d_fn", _hf_style_kernel_fallback(_torch_causal_conv))
    assert max_diff_vs_alone() > 1e-3


def test_packed_seq_installer_mixer_return_types():
    # Qwen3.5 does `h = linear_attn(...)` but `h, _ = self_attn(...)`: the
    # wrapper must hand each call site back the shape the mixer itself returns.
    hidden_dim = 8
    sample_lens = (3, 5, 4)
    torch.manual_seed(0)
    model = PackedHybridModel(hidden_dim)
    x = torch.randn(1, sum(sample_lens), hidden_dim)
    packed_kwargs = _packed_hf_kwargs(sample_lens, x.device)
    gdn = model.layers[0].linear_attn
    attn = model.layers[1].self_attn
    with torch.no_grad():
        ref_gdn = gdn(hidden_states=x, cache_params=None, **packed_kwargs)
        ref_attn = attn(hidden_states=x, past_key_values=None, **packed_kwargs)
        ref_model = model(x, **packed_kwargs)
        with _packed_mixers_installed(model, token_budget=6):
            tiled_gdn = gdn(hidden_states=x, cache_params=None, **packed_kwargs)
            tiled_attn = attn(hidden_states=x, past_key_values=None, **packed_kwargs)
            tiled_model = model(x, **packed_kwargs)
    assert torch.is_tensor(ref_gdn) and torch.is_tensor(tiled_gdn), type(tiled_gdn)
    assert isinstance(tiled_attn, tuple) and len(tiled_attn) == len(ref_attn) == 2, type(tiled_attn)
    assert tiled_attn[1] is None
    torch_assert_close(ref_gdn, tiled_gdn)
    torch_assert_close(ref_attn[0], tiled_attn[0])
    torch_assert_close(ref_model, tiled_model)


def test_packed_seq_installer_refuses_non_resetting_gdn(monkeypatch):
    hidden_dim = 8
    sample_lens = (3, 5, 4)
    this_module = sys.modules[__name__]
    cases = [
        ("torch_chunk_gated_delta_rule", _hf_style_kernel_fallback(_torch_decay_scan), "recurrent state.*cu_seqlens"),
        ("causal_conv1d_fn", _hf_style_kernel_fallback(_torch_causal_conv), "short-conv state.*seq_idx"),
        ("torch_chunk_gated_delta_rule", _decay_scan, "cannot be verified"),
    ]
    for name, kernel, match in cases:
        with monkeypatch.context() as patch:
            patch.setattr(this_module, name, kernel)
            with pytest.raises(RuntimeError, match=match):
                with _packed_mixers_installed(PackedHybridModel(hidden_dim), token_budget=6):
                    pass

    model = PackedHybridModel(hidden_dim)
    x = torch.randn(1, sum(sample_lens), hidden_dim)
    packed_kwargs = _packed_hf_kwargs(sample_lens, x.device)
    without_seq_idx = dict(packed_kwargs)
    without_seq_idx.pop("seq_idx")
    wrong_seq_idx = dict(packed_kwargs)
    wrong_seq_idx["seq_idx"] = torch.zeros_like(packed_kwargs["seq_idx"])
    with torch.no_grad(), _packed_mixers_installed(model, token_budget=6):
        with pytest.raises(RuntimeError, match="needs seq_idx"):
            model(x, **without_seq_idx)
        with pytest.raises(RuntimeError, match="seq_idx disagrees"):
            model(x, **wrong_seq_idx)


@pytest.mark.parametrize("sample_lens,token_budget", [
    ((4, 4, 4), 8),
    ((3, 7, 2, 4), 7),
    ((6, ), 3),
])
@pytest.mark.parametrize("zero_stage", [2, 3])
class TestPackedSeqTiledRematerializeGDN(DistributedTest):
    world_size = 1

    def test_packed_gdn_tiled_rematerialize_parity(self, zero_stage, sample_lens, token_budget):
        from deepspeed.runtime.activation_checkpointing import tiled_rematerialization as tiled_remat_mod

        config_dict, dtype = _zero_config(zero_stage, micro_batch=1, dtype=torch.float32)
        seed = 11
        hidden_dim = 16
        torch.manual_seed(seed)
        x = torch.rand((1, sum(sample_lens), hidden_dim), dtype=dtype)

        def build_engine():
            torch.manual_seed(seed)
            model = PackedHybridModel(hidden_dim).to(dtype)
            if zero_stage == 3:
                # Same leafing the bench applies to GatedDeltaNet classes.
                deepspeed.utils.set_z3_leaf_modules(model, [PackedGDNMix])
            engine, _, _, _ = deepspeed.initialize(config=config_dict,
                                                   model=model,
                                                   model_parameters=model.parameters())
            return engine

        model_a = build_engine()
        packed_kwargs = _packed_hf_kwargs(sample_lens, model_a.device)
        x_a = x.to(model_a.device).clone().detach().requires_grad_(True)
        loss_a = model_a(x_a, **packed_kwargs).square().mean()
        model_a.backward(loss_a)
        grads_a = {}
        for name, param in model_a.module.named_parameters():
            grads_a[name] = get_grad(param, zero_stage).clone()

        model_b = build_engine()
        x_b = x.to(model_b.device).clone().detach().requires_grad_(True)
        ready_flags = []
        orig_mark = tiled_remat_mod._mark_compute_params_ready

        def spy_mark(compute_params, is_last):
            ready_flags.append(bool(is_last))
            orig_mark(compute_params, is_last)

        tiled_remat_mod._mark_compute_params_ready = spy_mark
        try:
            with _packed_mixers_installed(model_b.module, token_budget=token_budget):
                loss_b = model_b(x_b, **packed_kwargs).square().mean()
                with CaptureStderr() as cs:
                    model_b.backward(loss_b)
        finally:
            tiled_remat_mod._mark_compute_params_ready = orig_mark
        assert "grad and param do not obey the gradient layout contract" not in cs.err, cs.err

        mixer_params = []
        for layer in model_b.module.layers:
            mixer = layer.linear_attn if hasattr(layer, "linear_attn") else layer.self_attn
            mixer_params.extend(mixer.parameters())
        _assert_compute_params_ready(mixer_params)
        # ds_grad_is_ready flips True only on each mixer call's last tile.
        num_tiles = len(plan_packed_tiles(packed_kwargs["cu_seq_lens_q"], token_budget=token_budget))
        num_mixer_calls = len(model_b.module.layers)
        if num_tiles == 1:
            assert ready_flags == [], ready_flags
        else:
            assert ready_flags == ([False] * (num_tiles - 1) + [True]) * num_mixer_calls, ready_flags

        torch_assert_close(loss_a, loss_b)
        torch_assert_close(x_a.grad, x_b.grad)
        for name, param in model_b.module.named_parameters():
            torch_assert_close(grads_a[name], get_grad(param, zero_stage), msg=name)


class BatchAndSequenceTiled(Module):
    """Decoder-style layer: batch remat outside, TiledMLP inside. The sweep failure."""

    def __init__(self, hidden_dim):
        super().__init__()
        self.mlp = SimpleMLP(hidden_dim)

    def forward(self, hidden_states):
        params = [p for p in self.mlp.parameters() if p.requires_grad]

        def fn(rows):
            return TiledMLP.apply(mlp_forward_orig, self.mlp, rows, 2, params)

        return tiled_rematerialize_compute(fn, hidden_states, 2, compute_params=params)


class TestBatchRematNestedTiledMLP(DistributedTest):
    world_size = 1

    def test_zero2_reduces_mlp_once(self):
        config_dict, dtype = _zero_config(2, micro_batch=2, dtype=torch.float32)
        hidden_dim = 16
        torch.manual_seed(1)
        model = BatchAndSequenceTiled(hidden_dim).to(dtype)
        engine, _, _, _ = deepspeed.initialize(config=config_dict, model=model, model_parameters=model.parameters())
        for _ in range(2):
            # requires_grad connects the loss to the tiled autograd Function.
            batch = torch.randn(2, 8, hidden_dim, device=engine.device, dtype=dtype, requires_grad=True)
            loss = engine(batch).square().mean()
            assert loss.grad_fn is not None
            engine.backward(loss)
            engine.step()


class PackedWithTiledMLP(Module):
    """Packed-sequence remat outside, TiledMLP inside. Sibling installs do not nest; this does."""

    def __init__(self, hidden_dim):
        super().__init__()
        self.proj = Linear(hidden_dim, hidden_dim, bias=False)
        self.mlp = SimpleMLP(hidden_dim)

    def forward(self, hidden_states, cu_seqlens):
        return self.mlp(self.proj(hidden_states))


class TestPackedTilingNestedTiledMLP(DistributedTest):
    world_size = 1

    def test_zero2_matches_untiled_and_reduces_once(self):
        config_dict, dtype = _zero_config(2, micro_batch=1, dtype=torch.float32)
        hidden_dim = 16
        sample_lens = (5, 5, 6)
        token_budget = 5
        total_tokens = sum(sample_lens)
        cu_seqlens = torch.tensor([0] + list(torch.tensor(sample_lens).cumsum(0)), dtype=torch.long)
        seed = 3

        def run(nest_tiled_mlp):
            torch.manual_seed(seed)
            model = PackedWithTiledMLP(hidden_dim).to(dtype)
            engine, _, _, _ = deepspeed.initialize(config=config_dict,
                                                   model=model,
                                                   model_parameters=model.parameters())
            if nest_tiled_mlp:
                mlp_params = [p for p in engine.module.mlp.parameters() if p.requires_grad]

                def tiled_forward(self, hidden_states, seqlens):

                    def fn(tile, cu_seqlens=None):
                        projected = self.proj(tile)
                        return TiledMLP.apply(mlp_forward_orig, self.mlp, projected, 2, mlp_params)

                    return packed_seq_tiled_rematerialize_compute(
                        fn,
                        hidden_states,
                        seqlens,
                        token_budget=token_budget,
                        compute_params=[self.proj.weight],
                    )

                engine.module.forward = tiled_forward.__get__(engine.module, PackedWithTiledMLP)
            rows = torch.randn(total_tokens, hidden_dim, device=engine.device, dtype=dtype, requires_grad=True)
            boundaries = cu_seqlens.to(engine.device)
            loss = engine(rows, boundaries).square().mean()
            assert loss.grad_fn is not None
            engine.backward(loss)
            grads = {name: get_grad(param, 2).detach().clone() for name, param in engine.module.named_parameters()}
            row_grad = rows.grad.detach().clone()
            _assert_compute_params_ready(list(engine.module.parameters()))
            engine.step()
            rows2 = torch.randn(total_tokens, hidden_dim, device=engine.device, dtype=dtype, requires_grad=True)
            loss2 = engine(rows2, boundaries).square().mean()
            engine.backward(loss2)
            engine.step()
            return loss.detach(), grads, row_grad

        loss_a, grads_a, row_grad_a = run(False)
        loss_b, grads_b, row_grad_b = run(True)
        torch_assert_close(loss_a, loss_b)
        torch_assert_close(row_grad_a, row_grad_b)
        for name in grads_a:
            torch_assert_close(grads_a[name], grads_b[name], msg=name)


@pytest.mark.parametrize("tile_forward", [False, True])
@pytest.mark.parametrize("batch_size,shards", [(2, 2), (3, 2), (2, 3), (5, 4), (1, 4)])
@pytest.mark.parametrize("zero_stage", [2, 3])
class TestTiledRematerialize(DistributedTest):
    world_size = 1

    def test_tiled_rematerialize_parity(self, zero_stage, batch_size, shards, tile_forward):
        config_dict, dtype = _zero_config(zero_stage, micro_batch=batch_size, dtype=torch.float32)
        seed = 42
        hidden_dim = 128
        seqlen = 17
        torch.manual_seed(seed)
        x = torch.rand((batch_size, seqlen, hidden_dim), dtype=dtype, requires_grad=True)

        torch.manual_seed(seed)
        model_a = StackedSeqMix(hidden_dim).to(dtype)
        model_a, _, _, _ = deepspeed.initialize(config=config_dict,
                                                model=model_a,
                                                model_parameters=model_a.parameters())
        x = x.to(model_a.device)
        x_a = x.clone().detach().requires_grad_(True)
        loss_a = model_a(x_a)
        model_a.backward(loss_a)
        grad_a1 = get_grad(model_a.module.mix1.proj.weight, zero_stage)
        grad_a2 = get_grad(model_a.module.mix2.proj.weight, zero_stage)
        x_grad_a = x_a.grad
        assert grad_a1 is not None and grad_a2 is not None and x_grad_a is not None

        def tiled_forward(self, x, mask=None, bias=None):
            compute_params = [self.proj.weight]
            return tiled_rematerialize(seqmix_forward_orig, self, x, shards, compute_params, tile_forward=tile_forward)

        torch.manual_seed(seed)
        SeqMix.forward = tiled_forward
        try:
            model_b = StackedSeqMix(hidden_dim).to(dtype)
            model_b, _, _, _ = deepspeed.initialize(config=config_dict,
                                                    model=model_b,
                                                    model_parameters=model_b.parameters())
            x_b = x.clone().detach().requires_grad_(True)
            loss_b = model_b(x_b)
            with CaptureStderr() as cs:
                model_b.backward(loss_b)
            assert "grad and param do not obey the gradient layout contract" not in cs.err, cs.err
            _assert_compute_params_ready([model_b.module.mix1.proj.weight, model_b.module.mix2.proj.weight])
            grad_b1 = get_grad(model_b.module.mix1.proj.weight, zero_stage)
            grad_b2 = get_grad(model_b.module.mix2.proj.weight, zero_stage)
            x_grad_b = x_b.grad

            torch_assert_close(loss_a, loss_b)
            torch_assert_close(grad_a1, grad_b1)
            torch_assert_close(grad_a2, grad_b2)
            torch_assert_close(x_grad_a, x_grad_b)

            # A stuck ds_grad_is_ready only shows up on the second step.
            model_a.step()
            model_b.step()
            x_a2 = x.clone().detach().requires_grad_(True)
            x_b2 = x.clone().detach().requires_grad_(True)
            loss_a2 = model_a(x_a2)
            model_a.backward(loss_a2)
            loss_b2 = model_b(x_b2)
            model_b.backward(loss_b2)
            _assert_compute_params_ready([model_b.module.mix1.proj.weight, model_b.module.mix2.proj.weight])
            torch_assert_close(get_grad(model_a.module.mix1.proj.weight, zero_stage),
                               get_grad(model_b.module.mix1.proj.weight, zero_stage))
        finally:
            SeqMix.forward = seqmix_forward_orig

    def test_tiled_rematerialize_compute_kwargs(self, zero_stage, batch_size, shards, tile_forward):
        config_dict, dtype = _zero_config(zero_stage, micro_batch=batch_size, dtype=torch.float32)
        seed = 7
        hidden_dim = 128
        seqlen = 9
        torch.manual_seed(seed)
        x = torch.rand((batch_size, seqlen, hidden_dim), dtype=dtype, requires_grad=True)
        mask = torch.ones((batch_size, seqlen), dtype=dtype)
        bias = torch.zeros((1, 1, hidden_dim), dtype=dtype)
        pos = (torch.rand((batch_size, seqlen, hidden_dim),
                          dtype=dtype), torch.rand((batch_size, seqlen, hidden_dim), dtype=dtype))

        torch.manual_seed(seed)
        model_a = SeqMix(hidden_dim).to(dtype)
        model_a, _, _, _ = deepspeed.initialize(config=config_dict,
                                                model=model_a,
                                                model_parameters=model_a.parameters())
        x = x.to(model_a.device)
        mask = mask.to(model_a.device)
        bias = bias.to(model_a.device)
        pos = (pos[0].to(model_a.device), pos[1].to(model_a.device))
        x_a = x.clone().detach().requires_grad_(True)
        loss_a = model_a(x_a, mask=mask, bias=bias, position_embeddings=pos).square().mean()
        model_a.backward(loss_a)
        grad_a = get_grad(model_a.module.proj.weight, zero_stage)

        torch.manual_seed(seed)
        model_b = SeqMix(hidden_dim).to(dtype)
        model_b, _, _, _ = deepspeed.initialize(config=config_dict,
                                                model=model_b,
                                                model_parameters=model_b.parameters())

        def model_forward(self, hidden, mask=None, bias=None, position_embeddings=None):
            return tiled_rematerialize_compute(
                lambda h, mask=None, bias=None, position_embeddings=None: seqmix_forward_orig(
                    self, h, mask=mask, bias=bias, position_embeddings=position_embeddings),
                hidden,
                shards,
                compute_params=[self.proj.weight],
                kwargs_to_shard=dict(mask=mask, position_embeddings=position_embeddings),
                kwargs_to_pass=dict(bias=bias),
                tile_forward=tile_forward,
            )

        # Replace only this engine's module forward.
        model_b.module.forward = model_forward.__get__(model_b.module, SeqMix)
        x_b = x.clone().detach().requires_grad_(True)
        loss_b = model_b(x_b, mask=mask, bias=bias, position_embeddings=pos).square().mean()
        with CaptureStderr() as cs:
            model_b.backward(loss_b)
        assert "grad and param do not obey the gradient layout contract" not in cs.err, cs.err
        _assert_compute_params_ready([model_b.module.proj.weight])
        torch_assert_close(loss_a, loss_b)
        torch_assert_close(grad_a, get_grad(model_b.module.proj.weight, zero_stage))
        torch_assert_close(x_a.grad, x_b.grad)

        with pytest.raises(TypeError, match="cu_seqlens"):
            tiled_rematerialize_compute(lambda h, cu_seqlens=None: h,
                                        x_b.detach(),
                                        shards,
                                        kwargs_to_shard=dict(cu_seqlens=torch.arange(batch_size +
                                                                                     1, device=x_b.device)))


@pytest.mark.parametrize("zero_stage", [2, 3])
class TestTiledRematerializeDistributedReduce(DistributedTest):
    world_size = 2

    def test_full_grad_matches_untiled(self, zero_stage):
        batch_size = 2
        shards = 3
        config_dict, dtype = _zero_config(zero_stage, micro_batch=batch_size, dtype=torch.float32)
        seed = 3
        hidden_dim = 128
        seqlen = 11
        torch.manual_seed(seed)
        x = torch.rand((batch_size, seqlen, hidden_dim), dtype=dtype, requires_grad=True)

        torch.manual_seed(seed)
        model_a = SeqMix(hidden_dim).to(dtype)
        model_a, _, _, _ = deepspeed.initialize(config=config_dict,
                                                model=model_a,
                                                model_parameters=model_a.parameters())
        x = x.to(model_a.device)
        x_a = x.clone().detach().requires_grad_(True)
        loss_a = model_a(x_a).square().mean()
        model_a.backward(loss_a)
        grad_a = get_grad(model_a.module.proj.weight, zero_stage)

        def tiled_forward(self, x, mask=None, bias=None):
            return tiled_rematerialize(seqmix_forward_orig, self, x, shards, [self.proj.weight])

        torch.manual_seed(seed)
        SeqMix.forward = tiled_forward
        try:
            model_b = SeqMix(hidden_dim).to(dtype)
            model_b, _, _, _ = deepspeed.initialize(config=config_dict,
                                                    model=model_b,
                                                    model_parameters=model_b.parameters())
            x_b = x.clone().detach().requires_grad_(True)
            loss_b = model_b(x_b).square().mean()
            model_b.backward(loss_b)
            _assert_compute_params_ready([model_b.module.proj.weight])
            torch_assert_close(loss_a, loss_b)
            torch_assert_close(grad_a, get_grad(model_b.module.proj.weight, zero_stage))
        finally:
            SeqMix.forward = seqmix_forward_orig

    def test_zero3_gathers_once_per_layer_not_per_row(self, zero_stage):
        if zero_stage != 3:
            pytest.skip("gather-once is a ZeRO-3 property")
        batch_size = 4
        shards = 4
        config_dict, dtype = _zero_config(zero_stage, micro_batch=batch_size, dtype=torch.float32)
        hidden_dim = 32
        seqlen = 8
        torch.manual_seed(0)
        model = SeqMix(hidden_dim).to(dtype)
        model, _, _, _ = deepspeed.initialize(config=config_dict, model=model, model_parameters=model.parameters())

        def tiled_forward(self, x, mask=None, bias=None):
            return tiled_rematerialize(seqmix_forward_orig, self, x, shards, [self.proj.weight])

        SeqMix.forward = tiled_forward
        try:
            from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
            weight = model.module.proj.weight
            gather_calls = [0]
            orig_all_gather = weight.all_gather
            orig_all_gather_coalesced = weight.all_gather_coalesced

            def _count_if_partitioned():
                if weight.ds_status == ZeroParamStatus.NOT_AVAILABLE:
                    gather_calls[0] += 1

            def counting_all_gather(param_list=None, async_op=False, hierarchy=0):
                _count_if_partitioned()
                return orig_all_gather(param_list=param_list, async_op=async_op, hierarchy=hierarchy)

            def counting_all_gather_coalesced(params, safe_mode=False, quantize=False):
                _count_if_partitioned()
                return orig_all_gather_coalesced(params, safe_mode=safe_mode, quantize=quantize)

            weight.all_gather = counting_all_gather
            weight.all_gather_coalesced = counting_all_gather_coalesced
            x = torch.rand((batch_size, seqlen, hidden_dim), dtype=dtype, device=model.device, requires_grad=True)
            loss = model(x).square().mean()
            model.backward(loss)
            # One gather for the forward tile loop, one for rematerialize. GAS would
            # gather on every row (shards times per phase).
            assert gather_calls[0] <= 2, gather_calls[0]
            assert gather_calls[0] < 2 * shards, gather_calls[0]
        finally:
            SeqMix.forward = seqmix_forward_orig

    def test_body_failure_is_not_masked_by_stale_active_accounting(self, zero_stage):
        """A failing tiled body must propagate its own exception.

        A child submodule hook can leave a stale ``ds_active_sub_modules`` entry behind. If the
        body then raises, ``GatheredParameters.__exit__`` force-partitions, ``free_param`` refuses
        the stale entry, and that RuntimeError replaces the original failure. In practice the
        original is an OOM, so the user is sent after a parameter-lifecycle bug they do not have.
        """
        if zero_stage != 3:
            pytest.skip("only ZeRO-3 force-partitions at the context boundary")
        config_dict, dtype = _zero_config(zero_stage, micro_batch=2, dtype=torch.float32)
        torch.manual_seed(0)
        model = SeqMix(32).to(dtype)
        model, _, _, _ = deepspeed.initialize(config=config_dict, model=model, model_parameters=model.parameters())

        # Deliberately not a RuntimeError, so the masking error cannot satisfy pytest.raises.
        class BodyFailure(Exception):
            pass

        weight = model.module.proj.weight
        with pytest.raises(BodyFailure):
            with _hold_gathered_params([weight]):
                weight.ds_active_sub_modules.add(1234)
                raise BodyFailure("stands in for an OOM inside the tile loop")
        assert not weight.ds_active_sub_modules


class _CaptureDeepSpeedWarnings:
    """Collect DeepSpeed logger warnings.

    The DeepSpeed logger sets ``propagate = False`` and writes to its own stdout
    handler, so neither ``caplog`` nor ``CaptureStderr`` sees these records.
    """

    def __init__(self):
        self.messages = []

    def __enter__(self):
        import logging

        from deepspeed.utils import logger

        capture = self

        class _Handler(logging.Handler):

            def emit(self, record):
                capture.messages.append(record.getMessage())

        self._handler = _Handler(level=logging.WARNING)
        self._logger = logger
        self._prior_level = logger.level
        logger.setLevel(logging.WARNING)
        logger.addHandler(self._handler)
        return self

    def __exit__(self, *exc):
        self._logger.removeHandler(self._handler)
        self._logger.setLevel(self._prior_level)
        return False

    @property
    def leaf_warnings(self):
        return [m for m in self.messages if "no leaf marking" in m]


def _run_tiled_stacked_seqmix(zero_stage, leaf):
    """One tiled forward/backward over two SeqMix layers, optionally leafed.

    Returns the captured leaf warnings. Two layers means the entry-point check
    runs twice per forward, so the one-shot guard is exercised by construction.
    """
    from deepspeed.runtime.activation_checkpointing import tiled_rematerialization as tiled_remat_mod
    from deepspeed.utils import set_z3_leaf_modules

    config_dict, dtype = _zero_config(zero_stage, micro_batch=2, dtype=torch.float32)
    hidden_dim = 16
    shards = 2

    def tiled_forward(self, x, mask=None, bias=None):
        return tiled_rematerialize(seqmix_forward_orig, self, x, shards, [self.proj.weight])

    torch.manual_seed(7)
    SeqMix.forward = tiled_forward
    try:
        model = StackedSeqMix(hidden_dim).to(dtype)
        if leaf:
            # Must precede deepspeed.initialize: the marking is read while ZeRO-3
            # registers its hooks.
            set_z3_leaf_modules(model, [SeqMix])
        model, _, _, _ = deepspeed.initialize(config=config_dict, model=model, model_parameters=model.parameters())
        x = torch.rand((2, 5, hidden_dim), dtype=dtype, device=model.device, requires_grad=True)

        tiled_remat_mod._warned_not_z3_leaf = False
        with _CaptureDeepSpeedWarnings() as cap:
            loss = model(x)
            model.backward(loss)
        assert torch.isfinite(loss), loss
        return cap.leaf_warnings
    finally:
        SeqMix.forward = seqmix_forward_orig
        tiled_remat_mod._warned_not_z3_leaf = False


class TestTiledRematerializeZ3LeafWarning(DistributedTest):
    """Tiling an unleafed ZeRO-3 module warns, since ZeRO-3 re-runs its per-submodule
    hooks once per shard and can register a param for reduction twice. Whether that
    reaches the ``ds_id`` assert in stage3 depends on bucketing -- these small models
    run clean -- so the contract is a warning, not a refusal."""

    world_size = 1

    def test_unleafed_zero3_warns_once(self):
        warnings = _run_tiled_stacked_seqmix(zero_stage=3, leaf=False)
        # Exactly one despite two tiled layers: the warning is one-shot per process.
        assert len(warnings) == 1, warnings
        assert "set_z3_leaf_modules" in warnings[0]
        assert "before" in warnings[0] and "deepspeed.initialize" in warnings[0]

    def test_leafed_zero3_does_not_warn(self):
        assert _run_tiled_stacked_seqmix(zero_stage=3, leaf=True) == []

    def test_zero2_does_not_warn(self):
        # Stage 2 params are not ZeRO-3 partitioned, so leafing does not apply.
        assert _run_tiled_stacked_seqmix(zero_stage=2, leaf=False) == []


def test_remat_offload_defers_to_outer_checkpoint_hooks():
    """Outer checkpoint hooks must drop the remat input. Offload would pin a second copy."""
    from deepspeed.runtime.activation_checkpointing import tiled_rematerialization as tiled

    calls = []

    class FakeEngine:

        def offload_input(self, tensor):
            calls.append(tuple(tensor.shape))
            return len(calls)

        def restore_input(self, token):
            raise AssertionError("checkpointed remat input must not come back from the offload engine")

    seen = []

    class SaveInput(torch.autograd.Function):

        @staticmethod
        def forward(ctx, hidden):
            ctx.ds_tiled_grad_enabled = True
            tiled._offload_remat_input(ctx, hidden)
            return hidden + 1

        @staticmethod
        def backward(ctx, grad):
            restored = tiled._load_remat_input(ctx)
            seen.append(restored.detach().clone())
            return grad

    saved_engine = tiled._activation_offload_engine
    tiled._activation_offload_engine = lambda enabled: FakeEngine()
    try:
        hidden = torch.randn(4, requires_grad=True)
        expected = hidden.detach().clone()
        packed = []

        def pack(tensor):
            packed.append(tensor.detach().clone())
            return len(packed) - 1

        def unpack(index):
            return packed[index]

        from deepspeed.runtime.activation_checkpointing.checkpointing import _suspend_tiled_remat_offload

        # An unrelated saved-tensor hook must not turn tiling offload off.
        # The engine-wide activation hook is one of those: it returns an
        # unmarked tensor unchanged.
        with torch.autograd.graph.saved_tensors_hooks(pack, unpack):
            SaveInput.apply(hidden)
        assert calls == [(4, )]

        calls.clear()
        packed.clear()
        seen.clear()
        hidden = torch.randn(4, requires_grad=True)
        expected = hidden.detach().clone()
        with _suspend_tiled_remat_offload(), torch.autograd.graph.saved_tensors_hooks(pack, unpack):
            out = SaveInput.apply(hidden)
        assert calls == []
        assert len(packed) == 1
        out.sum().backward()
        torch_assert_close(seen[0], expected)

        calls.clear()
        bare = torch.randn(4, requires_grad=True)

        class SaveInputBare(torch.autograd.Function):

            @staticmethod
            def forward(ctx, x):
                ctx.ds_tiled_grad_enabled = True
                tiled._offload_remat_input(ctx, x)
                return x + 1

            @staticmethod
            def backward(ctx, grad):
                return grad

        SaveInputBare.apply(bare)
        assert calls == [(4, )]
    finally:
        tiled._activation_offload_engine = saved_engine


def test_offload_restore_handle_does_not_pin_backward_hook_base():
    """A ZeRO-3 leaf module's full-backward hook hands ``forward`` a view whose ``_base`` is the
    caller's hidden. Holding the restore handle strongly would pin that base for the whole step and
    reclaim nothing, so assert the handle stays weak."""
    import gc
    import weakref
    from types import SimpleNamespace

    from deepspeed.runtime.activation_checkpointing import tiled_rematerialization as tiled

    class FakeEngine:

        def offload_input(self, tensor):
            return tensor.detach().clone()

        def restore_input(self, token):
            return token

    engine = FakeEngine()
    ctx = SimpleNamespace(ds_tiled_grad_enabled=True, ds_offload_engine=None, ds_offload_token=None)
    received = []

    class Hooked(Module):

        def forward(self, hidden):
            received.append(weakref.ref(hidden))
            tiled._offload_remat_input(ctx, hidden)
            return hidden * 2

    module = Hooked()
    module.register_full_backward_hook(lambda *unused: None)

    saved_engine = tiled._activation_offload_engine
    tiled._activation_offload_engine = lambda enabled: engine
    try:
        source = torch.randn(8, requires_grad=True)
        caller = source * 2
        expected = caller.detach().clone()
        caller_ref = weakref.ref(caller)
        module(caller)

        # forward got a hook view, not the caller's object
        assert received[0]() is not caller_ref()
        del caller
        gc.collect()
        assert caller_ref() is None, "backward-hook base is still pinned; restore handle is not weak"

        restored = tiled._load_remat_input(ctx)
        torch_assert_close(restored, expected)
        assert restored.requires_grad
    finally:
        tiled._activation_offload_engine = saved_engine


class TestTiledRematerializeNativeOffload(DistributedTest):
    world_size = 1

    def test_cpu_offload_empties_input_and_matches_grad(self):
        from deepspeed.accelerator import get_accelerator
        from deepspeed.runtime.activation_checkpointing import checkpointing as ds_ckpt

        if not get_accelerator().is_available() or get_accelerator().is_synchronized_device():
            pytest.skip("native activation offload needs an async accelerator")

        hidden_dim = 64
        batch_size = 2
        shards = 2
        seqlen = 16
        config_dict, dtype = _zero_config(2, micro_batch=batch_size, dtype=torch.float32)
        seed = 3
        ds_ckpt.configure(mpu_=None, checkpoint_in_cpu=True)

        def tiled_forward(self, x, mask=None, bias=None, position_embeddings=None):
            return tiled_rematerialize(seqmix_forward_orig, self, x, shards, [self.proj.weight])

        SeqMix.forward = tiled_forward
        try:
            torch.manual_seed(seed)
            model = StackedSeqMix(hidden_dim).to(dtype)
            model, _, _, _ = deepspeed.initialize(config=config_dict, model=model, model_parameters=model.parameters())
            x = torch.rand((batch_size, seqlen, hidden_dim), dtype=dtype, device=model.device, requires_grad=True)
            loss = model(x)
            engine = ds_ckpt._get_cpu_offload_engine()
            assert engine is not None
            assert engine.stats.offloaded_tensors + engine.stats.kept_last_tensors >= 1
            # Two layers: leaf is emptied (0-dim placeholder); keep_last holds the last hidden.
            # torch.empty([]) is 0-dim, so numel()==1; check shape, not numel.
            assert x.shape == torch.Size([]), x.shape
            model.backward(loss)
            assert x.grad is not None
            assert x.grad.shape == (batch_size, seqlen, hidden_dim)
            assert not engine._tracker
            assert not engine._keep_last
            assert not engine._fwd_stash
            assert not engine._reloads
        finally:
            SeqMix.forward = seqmix_forward_orig
            ds_ckpt._reset_cpu_offload_engine()
            ds_ckpt.configure(mpu_=None, checkpoint_in_cpu=False)

    def test_packed_tiling_offload_matches_native_host_footprint(self):
        """Packed tiling inside native offload must not pin extra host tensors.

        The outer non-reentrant checkpoint already offloads each layer input.
        Tiling recomputes the mixer from that checkpoint, so its saved hidden
        has to go through the checkpoint hook. A direct offload shows up as
        extra offloaded_tensors and extra host bytes.
        """
        from deepspeed.accelerator import get_accelerator
        from deepspeed.runtime.activation_checkpointing import checkpointing as ds_ckpt

        if not get_accelerator().is_available():
            pytest.skip("native activation offload needs an accelerator")

        # CPU has no side stream, and the engine refuses synchronized devices.
        # Install a blocking engine that still accounts host bytes so this
        # comparison runs without a GPU. CUDA keeps the real engine.
        cpu_offload = get_accelerator().is_synchronized_device()
        saved_get_engine = ds_ckpt._get_cpu_offload_engine
        saved_stream = None
        if cpu_offload:
            from deepspeed.runtime.activation_checkpointing.offload_activations import _ActivationOffloadEngine

            class _NoopStream:

                def wait_stream(self, other):
                    return None

                def wait_event(self, event):
                    return None

                def synchronize(self):
                    return None

                def record_event(self):
                    return None

            acc_cls = type(get_accelerator())
            saved_stream = acc_cls.Stream
            acc_cls.Stream = property(lambda self: (lambda *args, **kwargs: _NoopStream()))
            engine = _ActivationOffloadEngine(use_pin_memory=False,
                                              use_streams=False,
                                              keep_last_count=1,
                                              min_offload_bytes=0)

            def _cpu_should_skip(tensor, _engine=engine):
                num_bytes = _engine._num_bytes(tensor)
                if num_bytes < _engine.min_offload_bytes or isinstance(tensor, torch.nn.Parameter):
                    return True
                if tensor.numel() > 1 and 0 in tensor.stride():
                    return True
                return False

            engine._should_skip = _cpu_should_skip
            ds_ckpt._cpu_offload_engine = engine

            def _cpu_engine():
                if not ds_ckpt.CPU_CHECKPOINT or ds_ckpt.PARTITION_ACTIVATIONS:
                    return None
                return ds_ckpt._cpu_offload_engine

            ds_ckpt._get_cpu_offload_engine = _cpu_engine

        hidden_dim = 16
        seqlen = 8
        token_budget = 4
        device = torch.device(get_accelerator().current_device_name())

        class Mixer(Module):

            def __init__(self):
                super().__init__()
                self.proj = Linear(hidden_dim, hidden_dim, bias=False)

            def forward(self, hidden_states):
                return self.proj(hidden_states)

        class Block(Module):

            def __init__(self):
                super().__init__()
                self.mixer = Mixer()
                self.out = Linear(hidden_dim, hidden_dim, bias=False)

            def forward(self, hidden):
                mixer = self.mixer
                out = self.out

                def body(layer_input):
                    return out(mixer(layer_input))

                return ds_ckpt.non_reentrant_checkpoint(body, hidden)

        class Stack(Module):

            def __init__(self):
                super().__init__()
                self.blocks = torch.nn.ModuleList([Block(), Block()])

            def forward(self, hidden):
                for block in self.blocks:
                    hidden = block(hidden)
                return hidden.square().mean()

        def host_footprint(engine):
            engine._sync_copy_streams()
            live_bytes = 0
            for tracked in engine._tracker.values():
                cpu_tensor = tracked[0]
                live_bytes += cpu_tensor.numel() * cpu_tensor.element_size()
            return {
                "offloaded_tensors": engine.stats.offloaded_tensors,
                "offloaded_bytes": engine.stats.offloaded_bytes,
                "live_host_bytes": live_bytes,
                "live_tensors": len(engine._tracker),
                "kept_on_device": len(engine._keep_last),
            }

        def install_packed_tiling(model, cu_seqlens, tile_calls):

            def bind(mixer):
                proj = mixer.proj

                def tiled_forward(hidden_states, _proj=proj):
                    flat = hidden_states.squeeze(0)

                    def fn(tile, **unused):
                        tile_calls.append(int(tile.shape[0]))
                        return _proj(tile.unsqueeze(0)).squeeze(0)

                    params = [p for p in _proj.parameters() if p.requires_grad]
                    out = packed_seq_tiled_rematerialize_compute(fn,
                                                                 flat,
                                                                 cu_seqlens,
                                                                 token_budget=token_budget,
                                                                 compute_params=params)
                    return out.unsqueeze(0)

                mixer.forward = tiled_forward

            for block in model.blocks:
                bind(block.mixer)

        def run(state, x_source, cu_seqlens, tile):
            model = Stack().to(device)
            model.load_state_dict(state)
            tile_calls = []
            if tile:
                install_packed_tiling(model, cu_seqlens, tile_calls)
            ds_ckpt._reset_cpu_offload_engine()
            hidden = x_source.detach().clone().requires_grad_(True)
            loss = model(hidden)
            engine = ds_ckpt._get_cpu_offload_engine()
            assert engine is not None
            after_fwd = host_footprint(engine)
            loss.backward()
            after_bwd = host_footprint(engine)
            grads = {name: param.grad.detach().clone() for name, param in model.named_parameters()}
            hidden_grad = hidden.grad.detach().clone()
            return after_fwd, after_bwd, grads, hidden_grad, tile_calls

        torch.manual_seed(7)
        seed_model = Stack()
        state = seed_model.state_dict()
        cu_seqlens = torch.tensor([0, token_budget, seqlen], dtype=torch.int32, device=device)
        x_source = torch.randn(1, seqlen, hidden_dim, device=device)
        ds_ckpt.configure(mpu_=None, checkpoint_in_cpu=True)
        try:
            native_fwd, native_bwd, native_grads, native_x_grad, native_calls = run(state, x_source, cu_seqlens, False)
            tiled_fwd, tiled_bwd, tiled_grads, tiled_x_grad, tiled_calls = run(state, x_source, cu_seqlens, True)
        finally:
            ds_ckpt._reset_cpu_offload_engine()
            ds_ckpt.configure(mpu_=None, checkpoint_in_cpu=False)
            ds_ckpt._get_cpu_offload_engine = saved_get_engine
            if saved_stream is not None:
                type(get_accelerator()).Stream = saved_stream

        # Backward remat walks token_budget tiles. A no-op wrapper would never record one.
        assert token_budget in tiled_calls, tiled_calls
        assert native_calls == []
        assert native_fwd["offloaded_tensors"] > 0, native_fwd
        assert tiled_fwd == native_fwd, (native_fwd, tiled_fwd)
        assert tiled_bwd == native_bwd, (native_bwd, tiled_bwd)
        torch_assert_close(tiled_x_grad, native_x_grad)
        for name in native_grads:
            torch_assert_close(tiled_grads[name], native_grads[name])

    def test_cpu_offload_keeps_shared_rope_for_next_layer(self):
        from deepspeed.accelerator import get_accelerator
        from deepspeed.runtime.activation_checkpointing import checkpointing as ds_ckpt

        if not get_accelerator().is_available() or get_accelerator().is_synchronized_device():
            pytest.skip("native activation offload needs an async accelerator")

        hidden_dim = 64
        batch_size = 2
        shards = 2
        seqlen = 16
        config_dict, dtype = _zero_config(2, micro_batch=batch_size, dtype=torch.float32)
        ds_ckpt.configure(mpu_=None, checkpoint_in_cpu=True)

        def tiled_compute(self, x, mask=None, bias=None, position_embeddings=None):

            def _fn(h, **kw):
                return seqmix_forward_orig(self, h, **kw)

            return tiled_rematerialize_compute(
                _fn,
                x,
                shards,
                [self.proj.weight],
                kwargs_to_shard={
                    "position_embeddings": position_embeddings,
                    "mask": mask,
                },
                kwargs_to_pass={"bias": bias},
            )

        class StackedWithRope(Module):

            def __init__(self):
                super().__init__()
                self.mix1 = SeqMix(hidden_dim)
                self.mix2 = SeqMix(hidden_dim)

            def forward(self, hidden, pos, mask, bias):
                hidden = self.mix1(hidden, mask=mask, bias=bias, position_embeddings=pos)
                hidden = self.mix2(hidden, mask=mask, bias=bias, position_embeddings=pos)
                return hidden.square().mean()

        torch.manual_seed(11)
        ref = StackedWithRope().to(dtype)
        ref, _, _, _ = deepspeed.initialize(config=config_dict, model=ref, model_parameters=ref.parameters())
        x_ref = torch.rand((batch_size, seqlen, hidden_dim), dtype=dtype, device=ref.device, requires_grad=True)
        pos = (torch.rand((batch_size, seqlen, hidden_dim), dtype=dtype, device=ref.device),
               torch.rand((batch_size, seqlen, hidden_dim), dtype=dtype, device=ref.device))
        mask = torch.ones((batch_size, seqlen), dtype=dtype, device=ref.device)
        bias = torch.zeros((1, 1, hidden_dim), dtype=dtype, device=ref.device)
        loss_ref = ref(x_ref, pos, mask, bias)
        ref.backward(loss_ref)
        grad_ref = get_grad(ref.module.mix1.proj.weight, 2)
        x_grad_ref = x_ref.grad

        SeqMix.forward = tiled_compute
        try:
            torch.manual_seed(11)
            model = StackedWithRope().to(dtype)
            model, _, _, _ = deepspeed.initialize(config=config_dict, model=model, model_parameters=model.parameters())
            x = x_ref.detach().clone().requires_grad_(True)
            loss = model(x, pos, mask, bias)
            engine = ds_ckpt._get_cpu_offload_engine()
            assert engine is not None
            assert engine.stats.offloaded_tensors + engine.stats.kept_last_tensors >= 1
            # Shared RoPE/mask/bias must stay full so the second layer can index dim 1.
            assert pos[0].shape == (batch_size, seqlen, hidden_dim), pos[0].shape
            assert pos[1].shape == (batch_size, seqlen, hidden_dim), pos[1].shape
            assert mask.shape == (batch_size, seqlen), mask.shape
            assert bias.shape == (1, 1, hidden_dim), bias.shape
            model.backward(loss)
            torch_assert_close(loss, loss_ref)
            torch_assert_close(get_grad(model.module.mix1.proj.weight, 2), grad_ref)
            torch_assert_close(x.grad, x_grad_ref)
            assert not engine._tracker
            assert not engine._keep_last
            assert not engine._fwd_stash
            assert not engine._reloads
        finally:
            SeqMix.forward = seqmix_forward_orig
            ds_ckpt._reset_cpu_offload_engine()
            ds_ckpt.configure(mpu_=None, checkpoint_in_cpu=False)


class FatSeqMix(Module):

    def __init__(self, hidden_dim, wide):
        super().__init__()
        self.up = Linear(hidden_dim, wide, bias=False)
        self.down = Linear(wide, hidden_dim, bias=False)

    def forward(self, x):
        return self.down(self.up(x).cumsum(dim=1))


fat_forward_orig = FatSeqMix.forward


class TestTiledRematerializePeak(DistributedTest):
    world_size = 1

    def test_batch2_tiled_peak_near_batch1(self):
        from deepspeed.accelerator import get_accelerator
        if not get_accelerator().is_available() or get_accelerator().device_name() == "cpu":
            pytest.skip("peak check needs an accelerator")

        hidden_dim = 64
        wide = 2048
        seqlen = 128
        config_dict, dtype = _zero_config(2, micro_batch=2)
        seed = 1

        def peak_for(batch, shards):
            torch.manual_seed(seed)
            model = FatSeqMix(hidden_dim, wide).to(dtype)
            cfg = dict(config_dict)
            cfg["train_micro_batch_size_per_gpu"] = batch
            model, _, _, _ = deepspeed.initialize(config=cfg, model=model, model_parameters=model.parameters())
            x = torch.rand((batch, seqlen, hidden_dim), dtype=dtype, device=model.device, requires_grad=True)
            if shards > 1:

                def _tiled(self, x):
                    params = [self.up.weight, self.down.weight]
                    return tiled_rematerialize(fat_forward_orig, self, x, shards, params)

                FatSeqMix.forward = _tiled
            else:
                FatSeqMix.forward = fat_forward_orig
            get_accelerator().empty_cache()
            get_accelerator().reset_peak_memory_stats()
            loss = model(x).square().mean()
            model.backward(loss)
            peak = get_accelerator().max_memory_allocated()
            FatSeqMix.forward = fat_forward_orig
            return peak

        peak_b2_untiled = peak_for(2, 1)
        peak_b2_tiled = peak_for(2, 2)
        # Forward is full-batch; rematerialize is tiled, so step peak must not
        # exceed an untiled B=2 pass (allocator slack).
        assert peak_b2_tiled <= peak_b2_untiled * 1.15 + 8 * 1024**2, (peak_b2_untiled, peak_b2_tiled)


def test_plan_chain_tiles():
    # A sequence that fits in the budget is one tile, even when it is not a multiple of the chunk.
    assert plan_chain_tiles(100, 200) == [(0, 100)]
    assert plan_chain_tiles(64, 64) == [(0, 64)]
    # Budget rounds down to a multiple of 64, and the tail keeps the remainder.
    assert plan_chain_tiles(200, 100) == [(0, 64), (64, 128), (128, 192), (192, 200)]
    assert plan_chain_tiles(256, 128) == [(0, 128), (128, 256)]
    assert plan_chain_tiles(65, 64) == [(0, 64), (64, 65)]
    # A budget below one chunk is raised to the chunk size.
    assert plan_chain_tiles(200, 30) == [(0, 64), (64, 128), (128, 192), (192, 200)]
    assert plan_chain_tiles(20, 8, align=1) == [(0, 8), (8, 16), (16, 20)]
    # Merging the shortest adjacent pair keeps a chain valid when ranks must agree.
    assert _merge_smallest_chain_pair([(0, 64), (64, 128), (128, 140)]) == [(0, 64), (64, 140)]
    with pytest.raises(ValueError, match="token_budget"):
        plan_chain_tiles(10, 0)


class ChainMix(Module):
    """Conv plus a token recurrence, so a split is exact only when both states are carried."""

    def __init__(self, hidden_dim, kernel=4):
        super().__init__()
        self.proj = Linear(hidden_dim, hidden_dim, bias=False)
        self.conv_weight = torch.nn.Parameter(torch.randn(hidden_dim, kernel) * 0.1)
        self.out_proj = Linear(hidden_dim, hidden_dim, bias=False)
        self.chain = False
        self.token_budget = 8

    def tile(self, x, recurrent, conv_state):
        mixed = self.proj(x).transpose(1, 2)
        conv_out, conv_state = _causal_conv_carry(mixed, self.conv_weight, None, "silu", conv_state)
        values = conv_out.transpose(1, 2).float()
        if recurrent is None:
            state = torch.zeros(x.shape[0], x.shape[-1], dtype=torch.float32, device=x.device)
        else:
            state = recurrent
        steps = []
        for t in range(values.shape[1]):
            state = 0.5 * state + values[:, t]
            steps.append(state)
        out = torch.stack(steps, dim=1).to(dtype=x.dtype)
        return self.out_proj(out), state, conv_state

    def forward(self, x):
        if not self.chain:
            return self.tile(x, None, None)[0]
        params = [p for p in self.parameters() if p.requires_grad]
        return sequential_chain_rematerialize(self.tile,
                                              x,
                                              token_budget=self.token_budget,
                                              compute_params=params,
                                              align=1)


def test_sequential_chain_matches_full_sequence():
    torch.manual_seed(0)
    hidden = 8
    seq = 20
    model_a = ChainMix(hidden)
    model_b = ChainMix(hidden)
    model_b.load_state_dict(model_a.state_dict())
    model_b.chain = True
    x = torch.randn(2, seq, hidden)
    x_a = x.clone().detach().requires_grad_(True)
    x_b = x.clone().detach().requires_grad_(True)
    loss_a = model_a(x_a).square().mean()
    loss_b = model_b(x_b).square().mean()
    loss_a.backward()
    loss_b.backward()
    torch_assert_close(loss_a, loss_b)
    torch_assert_close(x_a.grad, x_b.grad)
    for (name, param_a), (_, param_b) in zip(model_a.named_parameters(), model_b.named_parameters()):
        torch_assert_close(param_a.grad, param_b.grad, msg=name)


@pytest.mark.parametrize("zero_stage", [2, 3])
class TestSequentialChainRematerialize(DistributedTest):
    world_size = 1

    def test_chain_grad_parity(self, zero_stage):
        from deepspeed.runtime.activation_checkpointing import tiled_rematerialization as tiled_remat_mod

        config_dict, dtype = _zero_config(zero_stage, micro_batch=1, dtype=torch.float32)
        seed = 3
        hidden = 8
        seq = 20
        torch.manual_seed(seed)
        x = torch.randn(1, seq, hidden, dtype=dtype)

        def build(chain):
            torch.manual_seed(seed)
            model = ChainMix(hidden).to(dtype)
            model.chain = chain
            if zero_stage == 3:
                deepspeed.utils.set_z3_leaf_modules(model, [ChainMix])
            engine, _, _, _ = deepspeed.initialize(config=config_dict,
                                                   model=model,
                                                   model_parameters=model.parameters())
            return engine

        model_a = build(False)
        x_a = x.to(model_a.device).clone().detach().requires_grad_(True)
        loss_a = model_a(x_a).square().mean()
        model_a.backward(loss_a)
        grads_a = {name: get_grad(param, zero_stage).clone() for name, param in model_a.module.named_parameters()}

        model_b = build(True)
        x_b = x.to(model_b.device).clone().detach().requires_grad_(True)
        ready_flags = []
        orig_mark = tiled_remat_mod._mark_compute_params_ready

        def spy_mark(compute_params, is_last):
            ready_flags.append(bool(is_last))
            orig_mark(compute_params, is_last)

        tiled_remat_mod._mark_compute_params_ready = spy_mark
        try:
            loss_b = model_b(x_b).square().mean()
            model_b.backward(loss_b)
        finally:
            tiled_remat_mod._mark_compute_params_ready = orig_mark

        tiles = plan_chain_tiles(seq, model_b.module.token_budget, align=1)
        assert ready_flags == [False] * (len(tiles) - 1) + [True], ready_flags
        _assert_compute_params_ready(model_b.module.parameters())
        torch_assert_close(loss_a, loss_b)
        torch_assert_close(x_a.grad, x_b.grad)
        for name, param in model_b.module.named_parameters():
            torch_assert_close(grads_a[name], get_grad(param, zero_stage), msg=name)


class PackedChainMix(ChainMix):
    """ChainMix over a packed row: every sample restarts from zero states, as a packed GDN does."""

    def forward(self, x, cu_seqlens):
        squeezed = x.dim() == 2
        if squeezed:
            x = x.unsqueeze(0)
        bounds = cu_seqlens.tolist()
        parts = [self.tile(x[:, start:end], None, None)[0] for start, end in zip(bounds[:-1], bounds[1:])]
        out = torch.cat(parts, dim=1)
        return out.squeeze(0) if squeezed else out


packedchainmix_forward_orig = PackedChainMix.forward

# (sample_lens, packed token_budget, chain token_budget, chain align, packed row is [T, H])
_NESTED_CHAIN_CASES = [
    # The oversized middle sample chains 8/8/4; the last packed tile is a plain multi-sample call.
    ((3, 20, 2, 4), 6, 8, 1, False),
    # The last packed tile is the chained sample, so readiness lands on its tile 0.
    ((3, 2, 20), 6, 8, 1, True),
    # Budget 100 rounds down to 64 and the tail is a short 22: cuts sit on chunk boundaries.
    ((5, 150, 3), 8, 100, 64, False),
]


@pytest.mark.parametrize("sample_lens,token_budget,chain_budget,chain_align,flat", _NESTED_CHAIN_CASES)
@pytest.mark.parametrize("zero_stage", [2, 3])
class TestPackedSeqNestedChainRematerialize(DistributedTest):
    world_size = 1

    def test_nested_chain_grad_parity(self, zero_stage, sample_lens, token_budget, chain_budget, chain_align, flat):
        from deepspeed.runtime.activation_checkpointing import tiled_rematerialization as tiled_remat_mod

        config_dict, dtype = _zero_config(zero_stage, micro_batch=1, dtype=torch.float32)
        seed = 5
        hidden = 8
        total_tokens = sum(sample_lens)
        cu_seqlens = torch.tensor([0] + list(torch.tensor(sample_lens).cumsum(0)), dtype=torch.long)
        torch.manual_seed(seed)
        x = torch.randn(total_tokens, hidden, dtype=dtype)
        if not flat:
            x = x.unsqueeze(0)

        def build():
            torch.manual_seed(seed)
            model = PackedChainMix(hidden).to(dtype)
            if zero_stage == 3:
                deepspeed.utils.set_z3_leaf_modules(model, [PackedChainMix])
            engine, _, _, _ = deepspeed.initialize(config=config_dict,
                                                   model=model,
                                                   model_parameters=model.parameters())
            return engine

        model_a = build()
        cu_seqlens = cu_seqlens.to(model_a.device)
        x_a = x.to(model_a.device).clone().detach().requires_grad_(True)
        loss_a = model_a(x_a, cu_seqlens).square().mean()
        model_a.backward(loss_a)
        grads_a = {name: get_grad(param, zero_stage).clone() for name, param in model_a.module.named_parameters()}

        model_b = build()

        def nested_forward(self, hidden_states, seqlens):
            params = [p for p in self.parameters() if p.requires_grad]
            return packed_seq_tiled_rematerialize_compute(
                lambda h, cu_seqlens=None: packedchainmix_forward_orig(self, h, cu_seqlens),
                hidden_states,
                seqlens,
                token_budget=token_budget,
                compute_params=params,
                chain_tile_fn=self.tile,
                chain_token_budget=chain_budget,
                chain_align=chain_align)

        model_b.module.forward = nested_forward.__get__(model_b.module, PackedChainMix)
        x_b = x.to(model_b.device).clone().detach().requires_grad_(True)
        ready_flags = []
        orig_mark = tiled_remat_mod._mark_compute_params_ready

        def spy_mark(compute_params, is_last):
            ready_flags.append(bool(is_last))
            orig_mark(compute_params, is_last)

        tiled_remat_mod._mark_compute_params_ready = spy_mark
        try:
            loss_b = model_b(x_b, cu_seqlens).square().mean()
            with CaptureStderr() as cs:
                model_b.backward(loss_b)
        finally:
            tiled_remat_mod._mark_compute_params_ready = orig_mark
        for failure in ("already been reduced", "Gradient computed twice"):
            assert failure not in cs.err, cs.err

        # One inner backward per chain tile of a chained sample, one per other packed tile.
        # ZeRO reduces on the first backward that sees the flag, so it must be the very last.
        tiles = plan_packed_tiles(cu_seqlens, token_budget=token_budget)
        inner_steps = []
        for start_sample, end_sample, start_tok, end_tok in tiles:
            steps = 1
            if end_sample - start_sample == 1:
                steps = len(plan_chain_tiles(end_tok - start_tok, chain_budget, align=chain_align))
            inner_steps.append(steps)
        assert max(inner_steps) > 1, inner_steps
        expected_flags = [False] * (sum(inner_steps) - 1) + [True]
        assert ready_flags == expected_flags, ready_flags
        _assert_compute_params_ready(model_b.module.parameters())

        torch_assert_close(loss_a, loss_b)
        torch_assert_close(x_a.grad, x_b.grad)
        for name, param in model_b.module.named_parameters():
            torch_assert_close(grads_a[name], get_grad(param, zero_stage), msg=name)


def test_nested_chain_rejects_token_kwargs():
    model = PackedChainMix(8)
    x = torch.randn(12, 8)
    cu_seqlens = torch.tensor([0, 2, 12])
    with pytest.raises(TypeError, match="token-major"):
        packed_seq_tiled_rematerialize_compute(lambda h, cu_seqlens=None, position_ids=None: h,
                                               x,
                                               cu_seqlens,
                                               token_budget=4,
                                               kwargs_to_token_shard=dict(position_ids=torch.arange(12)),
                                               chain_tile_fn=model.tile,
                                               chain_token_budget=4)


_FAKE_QWEN35_KERNELS = "_fake_qwen35_gdn_kernels"


def _fake_gated_delta_rule(query,
                           key,
                           value,
                           g=None,
                           beta=None,
                           initial_state=None,
                           output_final_state=False,
                           use_qk_l2norm_in_kernel=False):
    """Token-recurrent stand-in with fla's call contract: exact at any cut, unlike the 64-chunk kernel."""
    batch, seq_len, heads, k_dim = key.shape
    state = initial_state
    if state is None:
        state = torch.zeros(batch, heads, k_dim, value.shape[-1], dtype=torch.float32, device=key.device)
    outputs = []
    for t in range(seq_len):
        decay = g[:, t].exp()[..., None, None]
        update = beta[:, t, :, None, None] * key[:, t, :, :, None].float() * value[:, t, :, None, :].float()
        state = decay * state + update
        outputs.append(torch.einsum("bhk,bhkv->bhv", query[:, t].float(), state))
    return torch.stack(outputs, dim=1).to(value.dtype), state


class _FakeGatedNorm(Module):

    def forward(self, x, gate):
        return x * torch.nn.functional.silu(gate)


class FakeQwen35GatedDeltaNet(Module):
    """The attributes ``qwen35_gdn_tile_forward`` reads from ``Qwen3_5GatedDeltaNet``, at toy size."""

    # The helper finds the scan on the class's module, as transformers swaps it there.
    __module__ = _FAKE_QWEN35_KERNELS

    def __init__(self, hidden_dim=8, heads=2, head_dim=4, kernel=4):
        super().__init__()
        self.num_k_heads = self.num_v_heads = heads
        self.head_k_dim = self.head_v_dim = head_dim
        self.key_dim = self.value_dim = heads * head_dim
        conv_dim = 2 * self.key_dim + self.value_dim
        self.in_proj_qkv = Linear(hidden_dim, conv_dim, bias=False)
        self.in_proj_z = Linear(hidden_dim, self.value_dim, bias=False)
        self.in_proj_b = Linear(hidden_dim, heads, bias=False)
        self.in_proj_a = Linear(hidden_dim, heads, bias=False)
        self.conv1d = torch.nn.Conv1d(conv_dim, conv_dim, kernel, groups=conv_dim, bias=False, padding=kernel - 1)
        self.activation = "silu"
        self.A_log = torch.nn.Parameter(torch.zeros(heads))
        self.dt_bias = torch.nn.Parameter(torch.ones(heads))
        self.norm = _FakeGatedNorm()
        self.out_proj = Linear(self.value_dim, hidden_dim, bias=False)


def test_qwen35_gdn_tile_forward_applies_padding_mask(monkeypatch):
    kernels = type(sys)(_FAKE_QWEN35_KERNELS)
    kernels.torch_chunk_gated_delta_rule = _fake_gated_delta_rule
    monkeypatch.setitem(sys.modules, _FAKE_QWEN35_KERNELS, kernels)
    torch.manual_seed(0)
    module = FakeQwen35GatedDeltaNet()
    x = torch.randn(2, 10, 8)
    mask = torch.ones(2, 10)
    mask[0, 7:] = 0
    mask[1, :2] = 0

    x_masked = x.clone().requires_grad_(True)
    out_masked = qwen35_gdn_tile_forward(module, x_masked, attention_mask=mask)[0]
    out_masked.square().sum().backward()
    # Qwen's apply_mask_to_padding_states is exactly this product before the projections.
    x_premasked = x.clone().requires_grad_(True)
    out_premasked = qwen35_gdn_tile_forward(module, x_premasked * mask[:, :, None])[0]
    out_premasked.square().sum().backward()
    torch_assert_equal(out_masked, out_premasked)
    torch_assert_equal(x_masked.grad, x_premasked.grad)
    assert x_masked.grad[mask == 0].abs().max() == 0
    unmasked = qwen35_gdn_tile_forward(module, x)[0]
    assert (unmasked - out_masked).abs().max() > 1e-3

    # A full-sequence mask handed to one tile is refused, not broadcast.
    with pytest.raises(ValueError, match="attention_mask shape"):
        qwen35_gdn_tile_forward(module, x[:, :4], attention_mask=mask)

    # The documented chain recipe: mask once, then chain. Same output as one masked call.
    with torch.no_grad():
        chained = sequential_chain_rematerialize(lambda x_tile, h, c: qwen35_gdn_tile_forward(module, x_tile, h, c),
                                                 x * mask[:, :, None],
                                                 token_budget=4,
                                                 align=1)
    torch_assert_close(chained, out_masked.detach())
