# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
"""What `BF16_Optimizer.step` does with a gradient norm it cannot clip from.

`get_global_norm_of_tensors` does not raise on a non-finite norm; it writes -1, and
`get_norm_with_moe_layers` returns -1 for the same reason. Handed that, the clip coefficient
`max_norm / (global_norm + eps)` is negative, so clipping flips the sign of every gradient
rather than shrinking it. A bare `assert all_groups_norm > 0.` stood in the way, which said
nothing about the cause and also rejected a norm of exactly zero: a micro-batch whose labels
are all masked produces one, and clipping from it is a no-op, not an error.

CPU-only: the guard and the clip arithmetic are local to one rank.
"""

import pytest
import torch

from deepspeed.runtime.bf16_optimizer import BF16_Optimizer
from deepspeed.runtime.utils import clip_tensors_by_global_norm, mask_nan_or_inf_with_val_inplace


def test_a_non_finite_norm_is_reported_as_minus_one():
    """The premise: this is the value the norm helpers hand `step`, not an exception."""
    for bad in (float("inf"), float("-inf"), float("nan")):
        norm = torch.tensor(bad)
        mask_nan_or_inf_with_val_inplace(norm, device=norm.device)
        assert norm.item() == -1.0


def test_clipping_from_minus_one_flips_every_gradient():
    """Why the guard has to exist at all, stated as the behaviour it prevents."""
    gradient = torch.full((4, ), 2.0)
    clip_tensors_by_global_norm(input_tensors=[gradient], max_norm=1.0, global_norm=-1.0)
    assert (gradient < 0).all(), "a negative clip coefficient negates the gradient"


class _Bf16Step:
    """The parts of BF16_Optimizer.step that reaching the guard needs, and nothing else."""

    step = BF16_Optimizer.step

    def __init__(self, gradient_value):
        self._gradient = torch.full((4, ), gradient_value)
        self.mpu = None
        self.norm_type = 2
        self.graph_harvesting = False
        self.has_moe_layers = False
        self.clip_grad = 1.0
        self._global_grad_norm = 0.
        self._uses_muon = False
        self.grad_acc_dtype = torch.float32
        self.fp32_groups_flat_partition = []
        self.fp32_groups_gradient_flat_partition = []
        self.optimizer = type("_Recorder", (), {"step": lambda self: None})()
        self.stepped = False

    def get_grads_for_norm(self, for_clipping=False):
        if for_clipping:
            return [self._gradient]
        return [self._gradient], {}

    def _lazy_init_hp_params_optimizer_state(self):
        pass

    def update_lp_params(self):
        self.stepped = True

    def clear_hp_grads(self):
        pass


def test_an_all_zero_gradient_step_is_not_an_error():
    """A fully masked micro-batch gives exactly zero gradients; the step must still run."""
    optimizer = _Bf16Step(0.0)

    optimizer.step()

    assert optimizer.stepped
    assert float(optimizer._global_grad_norm) == 0.0
    assert (optimizer._gradient == 0).all(), "nothing to clip, and nothing was clipped"


def test_a_non_finite_norm_raises_with_a_reason():
    optimizer = _Bf16Step(float("inf"))

    with pytest.raises(RuntimeError, match="not finite"):
        optimizer.step()

    assert not optimizer.stepped
    assert (optimizer._gradient > 0).all(), "the gradients were not negated on the way out"


def test_an_ordinary_norm_still_clips():
    optimizer = _Bf16Step(2.0)  # norm 4.0 against a max of 1.0

    optimizer.step()

    assert float(optimizer._global_grad_norm) == pytest.approx(4.0)
    assert optimizer._gradient[0].item() == pytest.approx(0.5, rel=1e-4)
