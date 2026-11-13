# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import torch

from deepspeed.module_inject.containers.gptneox import DS_GPTNEOXContainer
from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus


class _RecordedGather:

    calls = []

    def __init__(self, params, modifier_rank=None):
        self.params = params
        self.modifier_rank = modifier_rank

    def __enter__(self):
        self.calls.append(self.modifier_rank)

    def __exit__(self, exc_type, exc_value, traceback):
        return False


def _partitioned_parameter(values):
    parameter = torch.nn.Parameter(values)
    parameter.ds_id = 0
    parameter.ds_status = ZeroParamStatus.NOT_AVAILABLE
    return parameter


def test_zero3_qkv_layout_round_trip_persists_mutation(monkeypatch):
    monkeypatch.setattr("deepspeed.runtime.zero.GatheredParameters", _RecordedGather)
    _RecordedGather.calls.clear()

    container = object.__new__(DS_GPTNEOXContainer)
    container.num_attention_heads = 2
    container.qkvw = _partitioned_parameter(torch.arange(48, dtype=torch.float32).reshape(12, 4))
    container.qkvb = _partitioned_parameter(torch.arange(12, dtype=torch.float32))

    initial_weight = container.qkvw.detach().clone()
    initial_bias = container.qkvb.detach().clone()

    container.transform_for_inference()
    assert not torch.equal(container.qkvw, initial_weight)
    assert not torch.equal(container.qkvb, initial_bias)

    container.transform_for_training()
    assert torch.equal(container.qkvw, initial_weight)
    assert torch.equal(container.qkvb, initial_bias)
    assert _RecordedGather.calls == [0, 0]
