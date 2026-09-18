# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
"""The Gram iteration has to survive a large but finite gradient.

`zeropower_via_gram_newtonschulz` normalizes its input by the Frobenius norm, which only
sets the scale: the iteration is scale free, so the orthogonal factor it returns must not
depend on how large the gradient was. Two ceilings broke that. The input was cast to the
compute dtype before it was normalized, and fp16 stops at 65504, so a finite fp32 gradient
arrived as inf. The norm itself squares every element, which overflows around 1.8e19 even
in fp32, so the divisor went infinite and the whole matrix went to zero.

Both are reachable from a real run: a loss scale of 1e4 on gradients of order 1 already
puts the input past the fp16 ceiling.
"""

import pytest
import torch

from deepspeed.accelerator import get_accelerator
from deepspeed.runtime.zero.muon.original_muon import (SingleDeviceMuon, ns_compute_dtype,
                                                       zeropower_via_gram_newtonschulz)


def _kernel():
    """The compiled wrapper hides the function under torch.compile; reach the original."""
    return getattr(zeropower_via_gram_newtonschulz, "_torchdynamo_orig_callable", zeropower_via_gram_newtonschulz)


@pytest.mark.parametrize("shape", [(32, 128), (128, 32), (2, 32, 128)])
@pytest.mark.parametrize("scale", [1e4, 1e5, 1e20, 1e30])
def test_gram_normalization_preserves_large_finite_gradient_direction(monkeypatch, shape, scale):
    # Exercise the fp16 arithmetic on CPU too, without requiring an fp16 training engine.
    monkeypatch.setattr(get_accelerator(), "is_fp16_supported", lambda: True)
    kernel = _kernel()
    gradient = torch.randn(shape, generator=torch.Generator().manual_seed(0))

    expected = kernel(gradient, steps=5)
    actual = kernel(gradient * scale, steps=5)

    assert expected.norm() > 0
    assert actual.dtype == ns_compute_dtype("gram")
    # fp16 carries about 3 decimal digits (eps 9.77e-4) and the iteration runs five steps,
    # so a few parts in a thousand is the arithmetic, not a scale dependence: a failure
    # here returns zeros or non-finite values, which these bounds are nowhere near.
    torch.testing.assert_close(actual, expected, atol=3e-3, rtol=2e-2)
    torch.testing.assert_close(kernel(torch.zeros_like(gradient), steps=5), torch.zeros_like(actual))


def test_single_device_muon_training_preserves_loss_scale(monkeypatch):
    """The same invariant through a real optimizer step rather than the kernel alone."""
    monkeypatch.setattr(get_accelerator(), "is_fp16_supported", lambda: True)
    generator = torch.Generator().manual_seed(0)
    initial = torch.randn(32, 128, generator=generator)
    inputs = torch.randn(8, 128, generator=generator)
    targets = torch.randn(8, 32, generator=generator)
    trained = []
    for scale in [1.0, 1e7, 1e25]:
        model = torch.nn.Linear(128, 32, bias=False)
        model.weight.data.copy_(initial)
        optimizer = SingleDeviceMuon(model.parameters(), lr=0.01)
        optimizer.zero_grad()
        loss = torch.nn.functional.mse_loss(model(inputs), targets)
        (loss * scale).backward()
        for group in optimizer.param_groups:
            group["lr"] = 0.01
        optimizer.step()
        trained.append(model.weight.detach())

    assert not torch.equal(trained[0], initial)
    for result in trained[1:]:
        torch.testing.assert_close(result, trained[0], atol=2e-4, rtol=1e-2)
