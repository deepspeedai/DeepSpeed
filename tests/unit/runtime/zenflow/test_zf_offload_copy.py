# Copyright (c) DeepSpeed Team.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
"""ZenFlow's offload copy must clear the gradient attribute it actually read.

`ZenFlowZeroOptimizerParallel.async_inplace_copy_grad_to_fp32_buffer_from_gpu` overrides the
base ZeRO-1/2 method only to write into the double-buffered `overlap_grad` instead of `.grad`.
Everything else has to match the base, and two things did not:

  * the source was fetched through `get_param_gradient_attribute`, which reads `param.grad_accum`
    when the gradient accumulation dtype differs from the parameter dtype, but the cleanup set
    `param.grad = None`. `_fill_param_grad_accum_attribute` adds into `grad_accum` when it is not
    None, so an uncleared buffer makes the next window accumulate on top of a gradient that has
    already been consumed.
  * the `grad_accum is None` guard called `.view()` on `None` in both branches.

These run the real method against a minimal stand-in, so they stay CPU-only and need no engine.
"""

import pytest
import torch

from deepspeed.runtime.zenflow.zenflow_stage_1_and_2 import ZenFlowZeroOptimizerParallel
from deepspeed.runtime.zero.stage_1_and_2 import DeepSpeedZeroOptimizer


class _Partition:

    def __init__(self, numel):
        self.overlap_grad = [torch.zeros(numel), torch.zeros(numel)]


class _Optimizer:
    """The attributes the method under test touches, and nothing else."""

    def __init__(self, use_grad_accum_attribute, numel):
        self.use_grad_accum_attribute = use_grad_accum_attribute
        self.grad_position = {0: [0, 0, 0, numel]}
        self.single_partition_of_fp32_groups = [_Partition(numel)]
        self.master_weights_and_grads_dtype = torch.float32

    def get_param_id(self, param):
        return 0

    def get_overlap_step_state(self):
        return 0

    get_param_gradient_attribute = DeepSpeedZeroOptimizer.get_param_gradient_attribute
    clear_grad_attribute = DeepSpeedZeroOptimizer.clear_grad_attribute
    copy = ZenFlowZeroOptimizerParallel.async_inplace_copy_grad_to_fp32_buffer_from_gpu


@pytest.mark.parametrize("use_grad_accum_attribute", [False, True])
def test_offload_copy_clears_the_attribute_it_read(use_grad_accum_attribute):
    param = torch.nn.Parameter(torch.zeros(4))
    param.grad = torch.full((4, ), 2.0)
    param.grad_accum = torch.full((4, ), 3.0) if use_grad_accum_attribute else None

    optimizer = _Optimizer(use_grad_accum_attribute, numel=4)
    optimizer.copy(param)

    # The gradient that was copied out is the one that gets cleared, so the next accumulation
    # window starts empty instead of adding to a gradient the optimizer has already taken.
    expected = 3.0 if use_grad_accum_attribute else 2.0
    assert torch.equal(optimizer.single_partition_of_fp32_groups[0].overlap_grad[0], torch.full((4, ), expected))
    assert optimizer.get_param_gradient_attribute(param) is None


def test_offload_copy_rejects_a_missing_gradient():
    """A missing gradient is an assertion, as in the base class, not an AttributeError on None."""
    param = torch.nn.Parameter(torch.zeros(4))
    param.grad = None
    optimizer = _Optimizer(use_grad_accum_attribute=False, numel=4)
    with pytest.raises(AssertionError):
        optimizer.copy(param)
