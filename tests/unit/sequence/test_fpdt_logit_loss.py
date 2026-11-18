# Copyright (c) DeepSpeed Team.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import pytest
import torch

import deepspeed.sequence.fpdt_layer as fpdt_layer
from deepspeed.sequence.fpdt_layer import FPDT_LogitsLoss


@pytest.mark.parametrize("loss_weights", ([1.0, 1.0], [1.0, 0.0]), ids=("unmasked", "zero_weight"))
def test_extreme_logits_match_cross_entropy(monkeypatch, loss_weights):
    monkeypatch.setattr(fpdt_layer.dist, "allgather_fn", lambda output, value, group=None: output.copy_(value))
    lm_output_data = torch.tensor([[[-1000.0, 1000.0]], [[1000.0, -1000.0]]])
    labels = torch.tensor([[0, 1]])
    reducer_weights = torch.tensor(loss_weights).unsqueeze(1)

    lm_output = lm_output_data.clone().requires_grad_()
    logit_weights = torch.eye(2, requires_grad=True)
    losses = FPDT_LogitsLoss.apply(lm_output, labels, logit_weights, 0, 1, None, 1)
    reduced_loss = (losses * reducer_weights).sum() / reducer_weights.sum()
    reduced_loss.backward()

    reference_lm_output = lm_output_data.clone().requires_grad_()
    reference_logit_weights = torch.eye(2, requires_grad=True)
    reference_logits = torch.matmul(reference_lm_output, reference_logit_weights.t())
    reference_losses = torch.nn.functional.cross_entropy(reference_logits.flatten(0, 1),
                                                         labels.t().reshape(-1),
                                                         reduction="none").reshape(2, 1)
    reference_reduced_loss = (reference_losses * reducer_weights).sum() / reducer_weights.sum()
    reference_reduced_loss.backward()

    assert losses.shape == (2, 1)
    torch.testing.assert_close(losses, torch.full_like(losses, 2000.0))
    torch.testing.assert_close(losses, reference_losses)
    torch.testing.assert_close(reduced_loss, reference_reduced_loss)
    torch.testing.assert_close(lm_output.grad, reference_lm_output.grad)
    torch.testing.assert_close(logit_weights.grad, reference_logit_weights.grad)
