# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
"""A zero-sized trainable parameter must survive a full ZeRO-3 step.

Four places treat "has elements" and "needs handling" as the same question:

* ``fetch_sub_module`` gates the all-gather on ``fetch_numel > 0``, so a submodule holding
  only zero-sized parameters is skipped, the parameters stay ``NOT_AVAILABLE``, and the
  wait loop immediately below asserts that they are ``AVAILABLE``;
* the prefetch submit gates on ``numel_prefetching > 0``, so an all-zero prefetch set is
  popped off the queue and then never prefetched (the fetch above still gathers those
  parameters when the submodule is reached, so this one costs the overlap, not the step);
* ``AllGatherCoalescedHandle.wait`` concatenates the per-rank slices, and no rank holds a
  slice of a zero-sized parameter, so ``torch.cat`` is handed an empty list;
* ``CUDAQuantizer.quantize`` derives its group count from ``numel``, so a zero-sized
  partition divides by zero.

Stages 1 and 2 are deliberately not covered here. #8280 and #8298 (issues #8279, #8297)
fixed the reduction path they were reported against, but this model shape still fails on
both from a different place — ``_update_model_bit16_weights`` drops a zero-element
parameter's shape when it repoints it at the flat buffer — which is a separate fix.
"""

import pytest
import torch

from unit.common import DistributedTest

import deepspeed

HIDDEN = 8


class EmptyTailModel(torch.nn.Module):
    """The shape from issue #8279: a trainable parameter with no elements, used in the loss."""

    def __init__(self, hidden=HIDDEN):
        super().__init__()
        self.dense = torch.nn.Linear(hidden, hidden, bias=False)
        self.empty = torch.nn.Linear(hidden, 0, bias=False)

    def forward(self, x):
        hidden = self.dense(x)
        # `empty(hidden)` is (batch, 0); summing it keeps the parameter in the autograd graph.
        return hidden.sum() + self.empty(hidden).sum()


class SharedModuleBlock(torch.nn.Module):
    """A sized and a zero-sized parameter on one module, so they are gathered together."""

    def __init__(self, hidden=HIDDEN):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.randn(hidden, hidden))
        self.empty = torch.nn.Parameter(torch.empty(0, hidden))

    def forward(self, x):
        return (x @ self.weight.t()).sum() + (x @ self.empty.t()).sum()


class OnlyEmptyBlock(torch.nn.Module):
    """A submodule whose parameters are *all* zero-sized."""

    def __init__(self, hidden=HIDDEN):
        super().__init__()
        self.first = torch.nn.Parameter(torch.empty(0, hidden))
        self.second = torch.nn.Parameter(torch.empty(0, hidden))

    def forward(self, x):
        return (x @ self.first.t()).sum() + (x @ self.second.t()).sum()


class SharedModuleModel(torch.nn.Module):
    """Several blocks, each owning a sized and a zero-sized parameter."""

    def __init__(self, hidden=HIDDEN, blocks=2):
        super().__init__()
        self.blocks = torch.nn.ModuleList([SharedModuleBlock(hidden) for _ in range(blocks)])

    def forward(self, x):
        return sum(block(x) for block in self.blocks)


class MixedModel(torch.nn.Module):
    """A sized submodule next to one holding only zero-sized parameters."""

    def __init__(self, hidden=HIDDEN, blocks=4):
        super().__init__()
        self.dense = torch.nn.Linear(hidden, hidden, bias=False)
        self.only_empty = torch.nn.ModuleList([OnlyEmptyBlock(hidden) for _ in range(blocks)])

    def forward(self, x):
        return self.dense(x).sum() + sum(block(x) for block in self.only_empty)


def _config(stage=3, **zero_overrides):
    return {
        "train_micro_batch_size_per_gpu": 1,
        "optimizer": {
            "type": "Adam",
            "params": {
                "lr": 1e-3
            }
        },
        "zero_optimization": {
            "stage": stage,
            **zero_overrides
        },
        "bf16": {
            "enabled": True
        },
    }


def _run_steps(model, config, steps=1):
    trainable = [p for p in model.parameters() if p.requires_grad]
    engine, *_ = deepspeed.initialize(model=model, model_parameters=trainable, config=config)
    for _ in range(steps):
        loss = engine(torch.randn(1, HIDDEN, device=engine.device, dtype=torch.bfloat16))
        engine.backward(loss)
        engine.step()
    return engine


class TestZeroSizedParameterSingleRank(DistributedTest):
    world_size = 1

    @pytest.mark.parametrize("stage", [3])
    def test_step_completes(self, stage):
        engine = _run_steps(EmptyTailModel(), _config(stage))
        assert engine.global_steps == 1


class TestZeroSizedParameterPartitioned(DistributedTest):
    world_size = 2

    @pytest.mark.parametrize("stage", [3])
    def test_step_completes(self, stage):
        # With more than one rank the stage-3 path goes through the real all-gather rather than
        # the single-rank shortcut, which is where the gate lives.
        engine = _run_steps(EmptyTailModel(), _config(stage))
        assert engine.global_steps == 1

    def test_coalesced_gather_completes(self):
        # One module owning both parameters puts them in a single coalesced gather, where
        # the zero-sized one contributes no slice to concatenate.
        engine = _run_steps(SharedModuleModel(), _config(), steps=2)

        assert engine.global_steps == 2
        block = engine.module.blocks[0]
        assert block.empty.shape == torch.Size([0, HIDDEN])
        # The gather has to hand back a dtype the module's own forward can use.
        assert block.empty.dtype == block.weight.dtype

    def test_completes_with_prefetch_enabled(self):
        # Submodules holding only zero-sized parameters, with prefetch on. This covers the
        # fetch gate under a prefetch configuration; it does not pin the prefetch submit
        # gate itself, which costs the prefetch rather than the gather.
        engine = _run_steps(MixedModel(), _config(stage3_prefetch_bucket_size=10000), steps=2)
        assert engine.global_steps == 2


class TestZeroSizedParameterQuantized(DistributedTest):
    world_size = 2

    def _skip_without_quantizer(self):
        from deepspeed.ops.op_builder import QuantizerBuilder
        if not deepspeed.ops.__compatible_ops__[QuantizerBuilder.NAME]:
            pytest.skip("QuantizerBuilder is not implemented")

    def test_quantized_weights(self):
        self._skip_without_quantizer()

        engine = _run_steps(MixedModel(), _config(zero_quantized_weights=True))

        assert engine.global_steps == 1

    def test_quantized_nontrainable_weights(self):
        self._skip_without_quantizer()
        model = MixedModel()
        for block in model.only_empty:
            block.first.requires_grad = False
            block.second.requires_grad = False

        engine = _run_steps(model, _config(zero_quantized_nontrainable_weights=True))

        assert engine.global_steps == 1
