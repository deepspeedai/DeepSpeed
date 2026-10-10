# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import torch

from deepspeed.runtime.zenflow.zenflow_stage_1_and_2 import ZenFlowZeroOptimizer


def test_average_tensor_without_reduce_scatter_passes_communication_dtype():
    optimizer = object.__new__(ZenFlowZeroOptimizer)
    optimizer.overlap_comm = False
    optimizer.reduce_scatter = False
    calls = []

    def gradient_reduction_w_predivide(tensor, communication_data_type):
        calls.append((tensor, communication_data_type))

    optimizer.gradient_reduction_w_predivide = gradient_reduction_w_predivide
    tensor = torch.zeros(4)

    optimizer.average_tensor(tensor, torch.bfloat16)

    assert calls == [(tensor, torch.bfloat16)]
