# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
"""ZeRO-3 leaves zero-element parameters unpartitioned.

Such a parameter has nothing to shard or gather, and it is already kept out of the optimizer
groups (``is_optimized_parameter``) and checkpointed with the frozen parameters. ZeRO-3 used to
partition it anyway, and then ``fetch_sub_module`` never gathered it: the gather is gated on the
number of elements left to fetch, so a submodule holding only zero-element parameters asserted
that they were still ``NOT_AVAILABLE``.
"""

import pytest
import torch

from unit.common import DistributedTest

import deepspeed
from deepspeed.runtime.zero.offload_config import OffloadStateTypeEnum
from deepspeed.runtime.zero.offload_states import get_state_devices
from deepspeed.runtime.zero.utils import zero_parameters
from deepspeed.utils.zero_to_fp32 import get_fp32_state_dict_from_zero_checkpoint

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
    """A sized and a zero-element parameter on one module."""

    def __init__(self, hidden=HIDDEN):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.randn(hidden, hidden))
        self.empty = torch.nn.Parameter(torch.empty(0, hidden))

    def forward(self, x):
        return (x @ self.weight.t()).sum() + (x @ self.empty.t()).sum()


class OnlyEmptyBlock(torch.nn.Module):
    """A submodule whose parameters all have zero elements."""

    def __init__(self, hidden=HIDDEN):
        super().__init__()
        self.first = torch.nn.Parameter(torch.empty(0, hidden))
        self.second = torch.nn.Parameter(torch.empty(0, hidden))

    def forward(self, x):
        return (x @ self.first.t()).sum() + (x @ self.second.t()).sum()


class SharedModuleModel(torch.nn.Module):

    def __init__(self, hidden=HIDDEN, blocks=2):
        super().__init__()
        self.blocks = torch.nn.ModuleList([SharedModuleBlock(hidden) for _ in range(blocks)])

    def forward(self, x):
        return sum(block(x) for block in self.blocks)


class MixedModel(torch.nn.Module):
    """A sized submodule next to several holding only zero-element parameters."""

    def __init__(self, hidden=HIDDEN, blocks=4):
        super().__init__()
        self.dense = torch.nn.Linear(hidden, hidden, bias=False)
        self.only_empty = torch.nn.ModuleList([OnlyEmptyBlock(hidden) for _ in range(blocks)])

    def forward(self, x):
        return self.dense(x).sum() + sum(block(x) for block in self.only_empty)


MODELS = {"empty_tail": EmptyTailModel, "shared_module": SharedModuleModel, "mixed": MixedModel}


def _config(**zero_overrides):
    return {
        "train_micro_batch_size_per_gpu": 1,
        "optimizer": {
            "type": "Adam",
            "params": {
                "lr": 1e-3
            }
        },
        "zero_optimization": {
            "stage": 3,
            **zero_overrides
        },
        "bf16": {
            "enabled": True
        },
    }


def _build(model_name, config, zero_init):
    if zero_init:
        with deepspeed.zero.Init(config_dict_or_path=config):
            model = MODELS[model_name]()
    else:
        model = MODELS[model_name]()
    engine, *_ = deepspeed.initialize(model=model, model_parameters=model.parameters(), config=config)
    return engine


def _train(engine, steps, input_requires_grad=False):
    for _ in range(steps):
        x = torch.randn(1, HIDDEN, device=engine.device, dtype=torch.bfloat16, requires_grad=input_requires_grad)
        loss = engine(x)
        engine.backward(loss)
        engine.step()


def _declared_numel(param):
    # A partitioned param's local data is an empty placeholder; ds_numel is its real size.
    return param.ds_numel if hasattr(param, "ds_numel") else param.numel()


def _empty_params(module):
    return {name: param for name, param in module.named_parameters() if _declared_numel(param) == 0}


def _assert_left_unpartitioned(engine):
    empty = _empty_params(engine.module)
    assert empty
    for name, param in empty.items():
        assert not hasattr(param, "ds_id"), name
        assert param.device == engine.device, name
        assert param.dtype == torch.bfloat16, name
    sized = [p for p in engine.module.parameters() if _declared_numel(p) > 0]
    assert sized and all(hasattr(p, "ds_id") for p in sized)
    # Post-init ZeRO-3 walks must skip unpartitioned empties; this helper is that contract.
    partitioned = list(zero_parameters(engine.module))
    assert set(partitioned) == set(sized)


class TestZeroElementParamSingleRank(DistributedTest):
    world_size = 1

    def test_step_completes(self):
        engine = _build("empty_tail", _config(), zero_init=False)
        _train(engine, steps=1)
        assert engine.global_steps == 1


class TestZeroElementParam(DistributedTest):
    world_size = 2

    @pytest.mark.parametrize("zero_init", [True, False])
    @pytest.mark.parametrize("model_name", list(MODELS))
    def test_left_unpartitioned_and_trains(self, model_name, zero_init):
        engine = _build(model_name, _config(), zero_init)
        _assert_left_unpartitioned(engine)

        _train(engine, steps=2)

        assert engine.global_steps == 2
        _assert_left_unpartitioned(engine)

    def test_with_prefetch(self):
        engine = _build("mixed", _config(stage3_prefetch_bucket_size=10000), zero_init=False)
        _train(engine, steps=2)
        assert engine.global_steps == 2

    def test_with_offload(self):
        config = _config(offload_param={"device": "cpu"}, offload_optimizer={"device": "cpu"})
        engine = _build("mixed", config, zero_init=True)
        _assert_left_unpartitioned(engine)
        _train(engine, steps=2)
        assert engine.global_steps == 2

    @pytest.mark.parametrize("zero_init", [True, False])
    def test_with_module_granularity_threshold(self, zero_init):
        engine = _build("mixed", _config(stage3_module_granularity_threshold=1000), zero_init)
        _train(engine, steps=2, input_requires_grad=True)
        assert engine.global_steps == 2

    def test_offload_and_reload_states(self):
        engine = _build("mixed", _config(), zero_init=False)
        _train(engine, steps=1)

        engine.offload_states()
        for state in (OffloadStateTypeEnum.lp_params, OffloadStateTypeEnum.hp_params,
                      OffloadStateTypeEnum.optim_states):
            assert get_state_devices(engine, state) == {torch.device("cpu")}, state
        engine.reload_states()
        for state in (OffloadStateTypeEnum.lp_params, OffloadStateTypeEnum.hp_params,
                      OffloadStateTypeEnum.optim_states):
            assert get_state_devices(engine, state) == {engine.device}, state

        _train(engine, steps=1)
        assert engine.global_steps == 2

    def test_checkpoint_round_trip(self, tmpdir):
        engine = _build("shared_module", _config(), zero_init=False)
        _train(engine, steps=1)
        engine.save_checkpoint(tmpdir)
        expected = {name: param.shape for name, param in _empty_params(engine.module).items()}

        consolidated = engine._zero3_consolidated_16bit_state_dict()
        if engine.global_rank == 0:
            assert {name: consolidated[name].shape for name in expected} == expected

        restored = _build("shared_module", _config(), zero_init=False)
        restored.load_checkpoint(tmpdir)
        _assert_left_unpartitioned(restored)
        _train(restored, steps=1)

        if engine.global_rank == 0:
            fp32 = get_fp32_state_dict_from_zero_checkpoint(tmpdir)
            assert {name: fp32[name].shape for name in expected} == expected


class TestZeroElementParamQuantized(DistributedTest):
    world_size = 2

    @pytest.mark.parametrize("quantize", ["zero_quantized_weights", "zero_quantized_nontrainable_weights"])
    def test_step_completes(self, quantize):
        from deepspeed.ops.op_builder import QuantizerBuilder
        if not deepspeed.ops.__compatible_ops__[QuantizerBuilder.NAME]:
            pytest.skip("QuantizerBuilder is not implemented")
        model = MixedModel()
        if quantize == "zero_quantized_nontrainable_weights":
            for block in model.only_empty:
                block.first.requires_grad = False
                block.second.requires_grad = False
        engine, *_ = deepspeed.initialize(model=model,
                                          model_parameters=model.parameters(),
                                          config=_config(**{quantize: True}))
        _train(engine, steps=1)
        assert engine.global_steps == 1
