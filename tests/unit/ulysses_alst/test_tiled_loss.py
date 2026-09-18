# Copyright (c) The DeepSpeed Contributors
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
"""
Arctic Long Sequence Training (ALST) TiledLoss component tests
"""

from deepspeed.runtime.sequence_parallel.ulysses_sp import TiledLoss
from unit.util import torch_assert_close
import pytest
import torch

# The tiled path and the reference average the cross entropy of the same non- -100 tokens and only
# differ in the order the terms are summed, so in float64 (eps 2.22e-16) the few extra additions can
# account for no more than about 1e-15 of relative error. 1e-12 keeps three orders of magnitude of
# headroom while staying far below what this test is built to catch, which are whole factors such as
# 0.5x or 1.5x on the gradient.
RTOL = 1e-12
ATOL = 1e-12


def token_mean_cross_entropy(logits, labels, vocab_size, shift_labels):
    """
    The token weighted mean cross entropy that `TiledLoss` is documented to reproduce. It keeps the
    dtype of `logits` so that the whole comparison can be run in float64.
    """
    return torch.nn.functional.cross_entropy(logits.reshape(-1, vocab_size),
                                             shift_labels.reshape(-1),
                                             ignore_index=-100,
                                             reduction="mean")


def tiled_and_reference(shift_labels, shards, vocab_size=7, seed=42):
    """
    Run `TiledLoss` and the plain whole sequence loss on the same logits and return both losses and
    both logits gradients.
    """
    torch.manual_seed(seed)
    shift_labels = torch.tensor([shift_labels], dtype=torch.long)
    logits = torch.randn((1, shift_labels.shape[1], vocab_size), dtype=torch.float64)

    logits_tiled = logits.clone().detach().requires_grad_(True)
    loss_tiled = TiledLoss.apply(token_mean_cross_entropy, logits_tiled, vocab_size, shift_labels, shards)
    loss_tiled.backward()

    logits_ref = logits.clone().detach().requires_grad_(True)
    loss_ref = token_mean_cross_entropy(logits_ref, None, vocab_size, shift_labels)
    loss_ref.backward()

    return loss_tiled, logits_tiled.grad, loss_ref, logits_ref.grad


@pytest.mark.parametrize(
    "shift_labels, shards",
    [
        ([1, 2, 3, 4, -100, -100, -100, -100], 2),
        ([1, 2, 3, 4, 5, -100, -100, 6], 2),
        ([1, 2, 3, -100, -100, -100, -100, -100], 4),
        ([1, 2, 3, 4, 5, 6, 0, 1], 4),
    ],
    ids=["fully_masked_shard", "uneven_token_counts", "more_shards_than_labels", "no_masked_labels"],
)
class TestTiledLoss:
    """
    `TiledLoss` documents itself as the token weighted mean of the cross entropy over the whole
    sequence, so both its value and its gradient have to match that reference. The label layouts
    cover a shard that is entirely -100, shards holding a different number of labels each, more shards
    than there are labels to spread over them, and, as a control, a sequence with no -100 at all,
    where every shard carries the same weight.
    """

    def test_tiled_loss_forward_matches_token_mean_cross_entropy(self, shift_labels, shards):
        loss_tiled, _, loss_ref, _ = tiled_and_reference(shift_labels, shards)
        torch_assert_close(loss_tiled, loss_ref, rtol=RTOL, atol=ATOL)

    def test_tiled_loss_gradient_matches_token_mean_cross_entropy(self, shift_labels, shards):
        _, grad_tiled, _, grad_ref = tiled_and_reference(shift_labels, shards)
        torch_assert_close(
            grad_tiled,
            grad_ref,
            rtol=RTOL,
            atol=ATOL,
            msg=lambda default: f"{default}\nthe tiled gradient is not the gradient of the tiled loss: "
            "each shard has to be weighted by its own share of the non- -100 labels, and fully masked "
            "shards must not take part in the average",
        )

    def test_tiled_loss_gradient_is_zero_on_masked_positions(self, shift_labels, shards):
        _, grad_tiled, _, _ = tiled_and_reference(shift_labels, shards)
        masked = torch.tensor([shift_labels], dtype=torch.long) == -100
        assert grad_tiled[masked].eq(0).all(), "positions with a -100 label must not receive a gradient"
