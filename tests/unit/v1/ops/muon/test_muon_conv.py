# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
"""Muon has to treat a convolution weight as the matrix it flattens it into.

`muon_update` reshapes a rank-4 weight to `[out, in * kh * kw]` before orthogonalizing it,
so every quantity derived from the gradient afterwards has to come from that matrix, not
from the kernel. Two things went the other way: the aspect-ratio scale read the gradient's
own trailing dimensions, which for a convolution are the kernel size, and the overflow
guard combined the flattened update with the rank-4 gradient.

So a convolution parameter received a different scale than the equivalent flattened
parameter, and the guard raised on the shape mismatch instead of discarding the step.

The parity checks below compare one `muon_update` call rather than a training trajectory.
The Newton-Schulz iteration runs in bfloat16 where the accelerator supports it, so
comparing two multi-step trajectories through it pins platform rounding rather than this
contract. A single call feeds the iteration a bit-identical matrix in both shapes.
"""

import pytest
import torch

from deepspeed.runtime.zero.muon.original_muon import (muon_update, zeropower_via_gram_newtonschulz,
                                                       zeropower_via_newtonschulz5)

OUT_CHANNELS, IN_CHANNELS, KERNEL = 32, 2, (3, 2)
FLAT_COLUMNS = IN_CHANNELS * KERNEL[0] * KERNEL[1]


def _conv_gradient(seed=0):
    torch.manual_seed(seed)
    return torch.randn(OUT_CHANNELS, IN_CHANNELS, *KERNEL)


@pytest.mark.parametrize("ns_method", ["standard", "gram"])
@pytest.mark.parametrize("nesterov", [True, False])
def test_convolution_update_matches_the_flattened_update(ns_method, nesterov):
    """A rank-4 weight and its own flattened copy have to receive the same update.

    Both go through `update.view(len(update), -1)`, so the iteration sees the same matrix
    and the results are equal exactly, with no tolerance to choose.
    """
    grad = _conv_gradient()
    # Compile each shape from scratch. Recompiling for the second shape makes its sizes symbolic,
    # and a scale computed at run time can round one bfloat16 step away from the folded constant.
    torch._dynamo.reset()
    conv = muon_update(grad.clone(), torch.zeros_like(grad), nesterov=nesterov, ns_method=ns_method)
    torch._dynamo.reset()
    flat = muon_update(grad.flatten(1).clone(),
                       torch.zeros(OUT_CHANNELS, FLAT_COLUMNS),
                       nesterov=nesterov,
                       ns_method=ns_method)

    assert conv.shape == grad.shape, "the update has to come back in the parameter's own shape"
    assert torch.equal(conv.flatten(1), flat), \
        f"convolution and flattened updates differ by {(conv.flatten(1) - flat).abs().max()}"


@pytest.mark.parametrize("ns_method", ["standard", "gram"])
def test_convolution_scale_comes_from_the_flattened_matrix(ns_method):
    """The aspect-ratio scale is read off [out, in*kh*kw], not off the kernel.

    For this weight the two readings are sqrt(max(1, 32/12)) = 1.633 and
    sqrt(max(1, 3/2)) = 1.225, which differ by a third. The iteration runs in bfloat16
    where it is supported (eps about 7.8e-3), so 2 percent is far above its rounding and
    far below the gap between the two candidates.
    """
    from_matrix = max(1, OUT_CHANNELS / FLAT_COLUMNS)**0.5
    from_kernel = max(1, KERNEL[0] / KERNEL[1])**0.5
    assert abs(from_matrix - from_kernel) > 0.3, "the fixture no longer separates the two readings"

    grad = _conv_gradient()
    momentum = torch.zeros_like(grad)
    update = muon_update(grad.clone(), momentum, nesterov=False, ns_method=ns_method)

    # `momentum` now holds what the iteration was handed, so orthogonalize it the same way.
    ns_fn = zeropower_via_gram_newtonschulz if ns_method == "gram" else zeropower_via_newtonschulz5
    orthogonal = ns_fn(momentum.flatten(1).clone(), steps=5).to(update.dtype)
    live = orthogonal.abs() > 1e-3
    applied = (update.flatten(1)[live] / orthogonal[live])

    torch.testing.assert_close(applied, torch.full_like(applied, from_matrix), rtol=2e-2, atol=0)


@pytest.mark.parametrize("ns_method", ["standard", "gram"])
def test_a_convolution_training_step_completes(ns_method):
    """The guard used to raise on the shape mismatch before the step could be applied."""
    from deepspeed.runtime.zero.muon.original_muon import SingleDeviceMuon

    torch.manual_seed(0)
    model = torch.nn.Conv2d(IN_CHANNELS, OUT_CHANNELS, kernel_size=KERNEL, bias=False)
    before = model.weight.detach().clone()
    optimizer = SingleDeviceMuon(model.parameters(), ns_method=ns_method)

    optimizer.zero_grad()
    torch.nn.functional.mse_loss(model(torch.randn(2, IN_CHANNELS, 5, 4)), torch.randn(2, OUT_CHANNELS, 3,
                                                                                       3)).backward()
    optimizer.step()

    assert model.weight.shape == before.shape
    assert torch.isfinite(model.weight).all(), "the step left the convolution weight non-finite"
    assert not torch.equal(model.weight.detach(), before), "the step did not move the weight"


@pytest.mark.parametrize("nesterov", [True, False])
def test_convolution_overflow_preserves_shape_and_momentum(nesterov):
    """The guard has to return the gradient's own shape so the step can be discarded."""
    grad = torch.ones(OUT_CHANNELS, IN_CHANNELS, *KERNEL)
    grad[0, 0, 0, 0] = float("inf")
    momentum = torch.ones_like(grad)
    before = momentum.clone()
    update = muon_update(grad, momentum, nesterov=nesterov)
    assert update.shape == grad.shape
    assert not torch.isfinite(update).all()
    torch.testing.assert_close(momentum, before)
