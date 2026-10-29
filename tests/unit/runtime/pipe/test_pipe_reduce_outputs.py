# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import torch

from deepspeed.runtime.pipe.engine import PipelineEngine


def _engine_stub():
    engine = PipelineEngine.__new__(PipelineEngine)
    torch.nn.Module.__init__(engine)
    engine.is_data_parallel = False
    return engine


def test_reduce_outputs_averages_each_loss_of_a_tuple():
    # A loss_fn may return several losses; eval_batch averages each over the micro-batches.
    outputs = [[torch.tensor(1.0), torch.tensor(10.0)], [torch.tensor(3.0), torch.tensor(30.0)]]
    reduced = _engine_stub()._reduce_outputs(outputs, micro_batches=len(outputs))
    assert [r.item() for r in reduced] == [2.0, 20.0]


def test_reduce_outputs_averages_a_single_loss():
    outputs = [torch.tensor(1.0), torch.tensor(3.0)]
    reduced = _engine_stub()._reduce_outputs(outputs, micro_batches=len(outputs))
    assert reduced.item() == 2.0
