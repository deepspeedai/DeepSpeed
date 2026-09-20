# Copyright (c) DeepSpeed Team.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
"""Per-head Muon on the accelerator: sharding equivalence, and end-to-end training.

Two concerns, in order:

1. **Why per-head survives a column-parallel split and the whole-matrix path does not.**
   Column-parallel tensor parallelism splits an attention projection on dim 0, which is the
   axis per-head Newton-Schulz batches over. Each rank therefore holds whole heads, and
   orthogonalizing them is the same computation whether the other ranks' heads are present or
   not. The whole-matrix path has no such property: orthogonalizing a block of rows is not the
   same as taking that block out of the orthogonalization of all of them.

2. **The whole path, running.** `deepspeed.initialize` tags the parameters, the ZeRO call
   sites carry the tag into `muon_update`, and a real training loop takes steps with it,
   across ZeRO stages and world size > 1.

See #8367. The arithmetic and the tagging are pinned on CPU in
`tests/unit/runtime/zero/test_per_head_muon.py`.
"""

from types import SimpleNamespace

import pytest
import torch

import deepspeed
from deepspeed.accelerator import get_accelerator
from deepspeed.runtime.zero.muon.original_muon import (
    _per_head_orthogonalize,
    zeropower_via_gram_newtonschulz,
    zeropower_via_newtonschulz5,
)
from unit.common import DistributedTest

# ---------------------------------------------------------------------------
# 1. A shard of the per-head result is the per-head result of the shard
# ---------------------------------------------------------------------------

HEADS, HEAD_DIM, HIDDEN, STEPS = 8, 32, 256, 5


@pytest.fixture
def grad():
    torch.manual_seed(0)
    return torch.randn(HEADS * HEAD_DIM, HIDDEN, device=get_accelerator().device_name())


def _relative(a, b):
    return ((a - b).norm() / b.norm()).item()


def _column_parallel_shards(tensor, tp):
    rows = tensor.shape[0] // tp
    return [tensor[r * rows:(r + 1) * rows].contiguous() for r in range(tp)]


@pytest.mark.parametrize("ns_method", ["gram", "standard"])
@pytest.mark.parametrize("tp", [2, 4])
def test_per_head_on_a_shard_is_the_shard_of_per_head(grad, ns_method, tp):
    """Exactly equal, not close: the shards are the same matrices in the same batch."""
    whole = _per_head_orthogonalize(grad.clone(), HEADS, STEPS, ns_method)
    sharded = torch.cat([
        _per_head_orthogonalize(shard.clone(), HEADS // tp, STEPS, ns_method)
        for shard in _column_parallel_shards(grad, tp)
    ])

    assert torch.equal(sharded, whole)


@pytest.mark.parametrize("ns_method", ["gram", "standard"])
def test_the_whole_matrix_path_does_not_survive_the_split(grad, ns_method):
    """The comparison this is measured against, so "exact" above means something.

    Newton-Schulz on a block of rows is a different computation from the same block of
    Newton-Schulz on every row, and the difference is not small.
    """
    ns_fn = zeropower_via_gram_newtonschulz if ns_method == "gram" else zeropower_via_newtonschulz5
    whole = ns_fn(grad.clone(), steps=STEPS)
    sharded = torch.cat([ns_fn(shard.clone(), steps=STEPS) for shard in _column_parallel_shards(grad, 2)])

    assert _relative(sharded, whole) > 0.1


def test_a_stale_head_count_splits_heads_in_half(grad):
    """What the tag does under tp=2 if it is not re-resolved against the shard.

    128 rows still divide by 8, so the divisibility check passes and each head is cut in two.
    """
    whole = _per_head_orthogonalize(grad.clone(), HEADS, STEPS, "gram")
    shard = _column_parallel_shards(grad, 2)[0]

    correct = _per_head_orthogonalize(shard.clone(), HEADS // 2, STEPS, "gram")
    stale = _per_head_orthogonalize(shard.clone(), HEADS, STEPS, "gram")

    assert torch.equal(correct, whole[:shard.shape[0]])
    assert _relative(stale, whole[:shard.shape[0]]) > 0.1


class TestPerHeadMuonUnderAutoTP(DistributedTest):
    """The tags AutoTP leaves behind, through a real `deepspeed.initialize`.

    `set_optimizer_flags` runs before `_configure_tensor_parallel`, and AutoTP replaces the
    parameter's `.data` in place, so the tag made against the whole model rides onto a shard.
    """
    world_size = 2

    def test_the_tags_describe_the_shard_and_not_the_model(self):
        transformers = pytest.importorskip("transformers")
        heads, head_dim = 8, 32
        config = transformers.LlamaConfig(hidden_size=heads * head_dim,
                                          num_attention_heads=heads,
                                          num_key_value_heads=heads,
                                          num_hidden_layers=2,
                                          intermediate_size=2 * heads * head_dim,
                                          vocab_size=128)
        model = transformers.AutoModelForCausalLM.from_config(config)

        engine, _, _, _ = deepspeed.initialize(model=model,
                                               model_parameters=model.parameters(),
                                               config={
                                                   "train_micro_batch_size_per_gpu": 1,
                                                   "gradient_accumulation_steps": 1,
                                                   "bf16": {
                                                       "enabled": True
                                                   },
                                                   "zero_optimization": {
                                                       "stage": 1
                                                   },
                                                   "tensor_parallel": {
                                                       "autotp_size": 2
                                                   },
                                                   "optimizer": {
                                                       "type": "Muon",
                                                       "params": {
                                                           "lr": 1e-3,
                                                           "per_head_muon": True
                                                       }
                                                   },
                                               })

        tagged = {n: p for n, p in model.named_parameters() if getattr(p, "muon_num_heads", None)}
        assert tagged, "AutoTP left nothing tagged"
        for name, param in tagged.items():
            assert param.shape[0] // param.muon_num_heads == head_dim, \
                f"{name}: {param.shape[0]} rows over {param.muon_num_heads} heads is not {head_dim} wide"
            assert param.muon_num_heads == heads // 2, \
                f"{name}: tp=2 leaves {heads // 2} heads on this rank, tagged {param.muon_num_heads}"

        # AutoTP asserts every rank in the TP group sees the same batch.
        ids = torch.arange(8, device=engine.device).unsqueeze(0) % 128
        out = engine(input_ids=ids, labels=ids)
        engine.backward(out.loss)
        engine.step()
        assert all(torch.isfinite(p).all() for p in model.parameters())


# ---------------------------------------------------------------------------
# 2. End-to-end training
# ---------------------------------------------------------------------------


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
