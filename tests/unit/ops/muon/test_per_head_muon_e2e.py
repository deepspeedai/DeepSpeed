# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
"""End-to-end training with per-head Muon, across ZeRO stages and world size > 1.

The unit tests next to this pin the arithmetic and the tagging. These run the whole path:
`deepspeed.initialize` tags the parameters, the ZeRO call sites carry the tag into
`muon_update`, and a real training loop takes steps with it. See #8367.
"""

from types import SimpleNamespace

import pytest
import torch

import deepspeed
from unit.common import DistributedTest


class AttentionModel(torch.nn.Module):
    """Small GQA-shaped model: split QKV, 8 query heads over 2 kv heads."""

    def __init__(self, hidden_dim=64, q_heads=8, kv_heads=2, head_dim=8, nlayers=2):
        super().__init__()
        self.q_heads, self.kv_heads, self.head_dim = q_heads, kv_heads, head_dim
        self.blocks = torch.nn.ModuleList()
        for _ in range(nlayers):
            self.blocks.append(
                torch.nn.ModuleDict({
                    "q_proj": torch.nn.Linear(hidden_dim, q_heads * head_dim, bias=False),
                    "k_proj": torch.nn.Linear(hidden_dim, kv_heads * head_dim, bias=False),
                    "v_proj": torch.nn.Linear(hidden_dim, kv_heads * head_dim, bias=False),
                    "o_proj": torch.nn.Linear(q_heads * head_dim, hidden_dim, bias=False),
                    "mlp": torch.nn.Linear(hidden_dim, hidden_dim, bias=False),
                }))
        self.cross_entropy_loss = torch.nn.CrossEntropyLoss()
        self.config = SimpleNamespace(num_attention_heads=q_heads,
                                      num_key_value_heads=kv_heads,
                                      hidden_size=hidden_dim,
                                      head_dim=head_dim)

    def forward(self, x, y):
        for b in self.blocks:
            q, k, v = b["q_proj"](x), b["k_proj"](x), b["v_proj"](x)
            rep = self.q_heads // self.kv_heads
            attn = q * k.repeat(1, rep) + v.repeat(1, rep)
            x = x + b["mlp"](b["o_proj"](attn))
        return self.cross_entropy_loss(x, y)


def _config(zero_stage, per_head, lr=0.01):
    return {
        "train_batch_size": 4,
        "optimizer": {
            "type": "muon",
            "params": {
                "lr": lr,
                "adam_lr": lr,
                "per_head_muon": per_head
            }
        },
        "zero_optimization": {
            "stage": zero_stage,
            # Muon does not support reduce-scatter; the existing Muon suite disables it the
            # same way (see TestMuonRejectsReduceScatter).
            "reduce_scatter": False,
        },
        "fp16": {
            "enabled": False
        },
        "bf16": {
            "enabled": True
        },
    }


def _train(model, config, steps=6, hidden_dim=64, seed=1234):
    engine, *_ = deepspeed.initialize(model=model, model_parameters=model.parameters(), config=config)
    tags = {n: getattr(p, "muon_num_heads", "MISSING") for n, p in model.named_parameters()}
    gen = torch.Generator().manual_seed(seed)
    losses = []
    for _ in range(steps):
        x = torch.randn(4, hidden_dim, generator=gen).to(engine.device).to(torch.bfloat16)
        y = torch.randint(0, hidden_dim, (4, ), generator=gen).to(engine.device)
        loss = engine(x, y)
        engine.backward(loss)
        engine.step()
        losses.append(loss.item())
    return tags, losses


@pytest.mark.parametrize("zero_stage", [1, 2, 3])
class TestPerHeadMuonEndToEnd(DistributedTest):
    world_size = 2

    def test_tags_reach_the_optimizer(self, zero_stage):
        """Per-parameter tags have to survive `deepspeed.initialize` into the ZeRO call sites."""
        torch.manual_seed(1234)
        tags, losses = _train(AttentionModel(), _config(zero_stage, per_head=True))

        assert tags["blocks.0.q_proj.weight"] == 8
        assert tags["blocks.0.k_proj.weight"] == 2, "GQA: kv projections carry the kv head count"
        assert tags["blocks.0.v_proj.weight"] == 2
        assert tags["blocks.0.o_proj.weight"] is None, "o_proj's heads are on the input axis"
        assert tags["blocks.0.mlp.weight"] is None
        assert all(torch.isfinite(torch.tensor(loss)) for loss in losses)

    def test_opt_in_is_off_by_default(self, zero_stage):
        torch.manual_seed(1234)
        tags, _ = _train(AttentionModel(), _config(zero_stage, per_head=False))

        assert all(v is None for v in tags.values()), {k: v for k, v in tags.items() if v is not None}

    def test_training_makes_progress_either_way(self, zero_stage):
        """Both paths have to train; this is the baseline delock asked for alongside per-head."""
        torch.manual_seed(1234)
        _, full = _train(AttentionModel(), _config(zero_stage, per_head=False))
        torch.manual_seed(1234)
        _, per_head = _train(AttentionModel(), _config(zero_stage, per_head=True))

        assert full[-1] < full[0], f"baseline did not train: {full}"
        assert per_head[-1] < per_head[0], f"per-head did not train: {per_head}"
