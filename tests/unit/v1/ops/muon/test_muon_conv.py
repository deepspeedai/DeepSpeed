# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
"""Muon has to treat a convolution weight as the matrix it flattens it into.

`muon_update` reshapes a rank-4 weight to `[out, in * kh * kw]` before orthogonalizing
it, so every quantity derived from the gradient afterwards has to come from that matrix,
not from the kernel. Two things went the other way: the aspect-ratio scale read the
gradient's own trailing dimensions, which for a convolution are the kernel size, and the
overflow guard combined the flattened update with the rank-4 gradient.

So a convolution parameter received a different scale than the equivalent flattened
parameter, and the guard raised on the shape mismatch instead of discarding the step.
"""

import pytest
import torch

from deepspeed.runtime.zero.muon.original_muon import muon_update


@pytest.mark.parametrize("ns_method", ["standard", "gram"])
def test_convolution_training_matches_flattened_muon(ns_method):
    """A Conv2d weight and its own flattened copy have to take the same step."""
    from deepspeed.runtime.zero.muon.original_muon import SingleDeviceMuon

    torch.manual_seed(0)
    model = torch.nn.Conv2d(2, 32, kernel_size=(3, 2), bias=False)
    flat_weight = torch.nn.Parameter(model.weight.detach().flatten(1).clone())
    optimizer = SingleDeviceMuon(model.parameters(), ns_method=ns_method)
    reference = SingleDeviceMuon([flat_weight], ns_method=ns_method)
    inputs = torch.randn(2, 2, 5, 4)
    targets = torch.randn(2, 32, 3, 3)

    for _ in range(2):
        optimizer.zero_grad()
        loss = torch.nn.functional.mse_loss(model(inputs), targets)
        loss.backward()
        reference.zero_grad()
        flat_weight.grad = model.weight.grad.detach().flatten(1).clone()
        optimizer.step()
        reference.step()
        torch.testing.assert_close(model.weight.flatten(1), flat_weight)


@pytest.mark.parametrize("nesterov", [True, False])
def test_convolution_overflow_preserves_shape_and_momentum(nesterov):
    """The guard has to return the gradient's own shape so the step can be discarded."""
    grad = torch.ones(32, 2, 3, 2)
    grad[0, 0, 0, 0] = float("inf")
    momentum = torch.ones_like(grad)
    before = momentum.clone()
    update = muon_update(grad, momentum, nesterov=nesterov)
    assert update.shape == grad.shape
    assert not torch.isfinite(update).all()
    torch.testing.assert_close(momentum, before)
