# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
"""CPU-only numerics tests for the distillation divergences.

These exercise the loss math without needing GPUs, models, or a torchrun
launcher. Run from the example root with::

    cd examples/opsd && python -m pytest tests/test_losses.py -v
"""

import pytest
import torch

from opsd.losses import chunked_distillation_loss, per_token_logprobs
from opsd.utils import build_response_mask, shift_for_next_token_prediction


@pytest.mark.parametrize("loss_type", ["forward_kl", "reverse_kl", "jsd"])
def test_zero_when_identical(loss_type):
    torch.manual_seed(0)
    logits = torch.randn(2, 8, 32)
    mask = torch.ones(2, 8)
    loss = chunked_distillation_loss(logits, logits.clone(), mask, loss_type=loss_type)
    assert loss.item() == pytest.approx(0.0, abs=1e-5)


@pytest.mark.parametrize("loss_type", ["forward_kl", "reverse_kl", "jsd"])
def test_positive_when_different(loss_type):
    torch.manual_seed(0)
    s = torch.randn(2, 8, 32)
    t = torch.randn(2, 8, 32)
    mask = torch.ones(2, 8)
    loss = chunked_distillation_loss(s, t, mask, loss_type=loss_type)
    assert loss.item() > 0.0


@pytest.mark.parametrize("loss_type", ["forward_kl", "reverse_kl", "jsd"])
def test_chunking_equivalent_to_unchunked(loss_type):
    torch.manual_seed(0)
    s = torch.randn(2, 100, 32)
    t = torch.randn(2, 100, 32)
    mask = torch.ones(2, 100)
    loss_chunked = chunked_distillation_loss(s, t, mask, loss_type=loss_type, chunk_size=10)
    loss_whole = chunked_distillation_loss(s, t, mask, loss_type=loss_type, chunk_size=10_000)
    assert torch.allclose(loss_chunked, loss_whole, atol=1e-5)


def test_mask_excludes_tokens():
    torch.manual_seed(0)
    s = torch.randn(2, 8, 32)
    t = torch.randn(2, 8, 32)
    half_mask = torch.tensor([[1, 1, 1, 1, 0, 0, 0, 0], [1, 1, 1, 1, 0, 0, 0, 0]], dtype=torch.float32)
    loss_direct = chunked_distillation_loss(s[:, :4], t[:, :4], torch.ones(2, 4), loss_type="reverse_kl")
    loss_masked = chunked_distillation_loss(s, t, half_mask, loss_type="reverse_kl")
    assert torch.allclose(loss_direct, loss_masked, atol=1e-5)


def test_gradient_flows_to_student():
    torch.manual_seed(0)
    s = torch.randn(2, 8, 32, requires_grad=True)
    t = torch.randn(2, 8, 32)
    mask = torch.ones(2, 8)
    loss = chunked_distillation_loss(s, t, mask, loss_type="reverse_kl")
    loss.backward()
    assert s.grad is not None
    assert s.grad.abs().sum().item() > 0


def test_gradient_does_not_flow_to_teacher_when_detached():
    torch.manual_seed(0)
    s = torch.randn(2, 8, 32, requires_grad=True)
    t = torch.randn(2, 8, 32, requires_grad=True)
    mask = torch.ones(2, 8)
    loss = chunked_distillation_loss(s, t.detach(), mask, loss_type="reverse_kl")
    loss.backward()
    assert t.grad is None


def test_unknown_loss_type_raises():
    s = torch.randn(2, 4, 8)
    t = torch.randn(2, 4, 8)
    mask = torch.ones(2, 4)
    with pytest.raises(ValueError, match="Unknown loss_type"):
        chunked_distillation_loss(s, t, mask, loss_type="totally_made_up")


def test_shape_mismatch_raises():
    s = torch.randn(2, 4, 8)
    t = torch.randn(2, 5, 8)
    mask = torch.ones(2, 4)
    with pytest.raises(ValueError, match="shape mismatch"):
        chunked_distillation_loss(s, t, mask)


def test_mask_shape_mismatch_raises():
    s = torch.randn(2, 4, 8)
    t = torch.randn(2, 4, 8)
    mask = torch.ones(2, 5)
    with pytest.raises(ValueError, match="does not match"):
        chunked_distillation_loss(s, t, mask)


@pytest.mark.parametrize("temperature", [0.5, 1.0, 2.0])
def test_temperature_changes_loss_but_stays_finite(temperature):
    torch.manual_seed(0)
    s = torch.randn(2, 8, 32)
    t = torch.randn(2, 8, 32)
    mask = torch.ones(2, 8)
    loss = chunked_distillation_loss(s, t, mask, loss_type="reverse_kl", temperature=temperature)
    assert torch.isfinite(loss).item()


def test_jsd_is_symmetric():
    torch.manual_seed(0)
    a = torch.randn(2, 8, 32)
    b = torch.randn(2, 8, 32)
    mask = torch.ones(2, 8)
    jsd_ab = chunked_distillation_loss(a, b, mask, loss_type="jsd")
    jsd_ba = chunked_distillation_loss(b, a, mask, loss_type="jsd")
    assert torch.allclose(jsd_ab, jsd_ba, atol=1e-5)


def test_all_zero_mask_returns_zero():
    torch.manual_seed(0)
    s = torch.randn(2, 8, 32)
    t = torch.randn(2, 8, 32)
    mask = torch.zeros(2, 8)
    loss = chunked_distillation_loss(s, t, mask, loss_type="reverse_kl")
    assert loss.item() == pytest.approx(0.0, abs=1e-6)


def test_per_token_logprobs_matches_manual():
    torch.manual_seed(0)
    logits = torch.randn(2, 4, 16)
    labels = torch.randint(0, 16, (2, 4))
    got = per_token_logprobs(logits, labels)
    expected = torch.log_softmax(logits.float(), dim=-1)
    expected = expected.gather(-1, labels.unsqueeze(-1)).squeeze(-1)
    assert torch.allclose(got, expected, atol=1e-6)


def test_build_response_mask_basic():
    attention_mask = torch.tensor([[1, 1, 1, 1, 0], [1, 1, 1, 1, 1]])
    response_start_idx = torch.tensor([2, 3])
    resp = build_response_mask(response_start_idx, attention_mask)
    expected = torch.tensor([[0, 0, 1, 1, 0], [0, 0, 0, 1, 1]])
    assert torch.equal(resp, expected)


def test_build_response_mask_validates_shapes():
    with pytest.raises(ValueError, match="response_start_idx must be 1-D"):
        build_response_mask(torch.zeros(2, 2), torch.ones(2, 4))
    with pytest.raises(ValueError, match="attention_mask must be 2-D"):
        build_response_mask(torch.zeros(2), torch.ones(4))
    with pytest.raises(ValueError, match="batch"):
        build_response_mask(torch.zeros(3), torch.ones(2, 4))


def test_shift_for_next_token_prediction_shapes():
    logits = torch.randn(2, 5, 8)
    labels = torch.randint(0, 8, (2, 5))
    sl, sla = shift_for_next_token_prediction(logits, labels)
    assert sl.shape == (2, 4, 8)
    assert sla.shape == (2, 4)
