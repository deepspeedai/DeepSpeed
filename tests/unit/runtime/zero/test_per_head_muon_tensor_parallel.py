# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
"""Per-head Muon under tensor parallelism. See #8367.

`set_optimizer_flags` runs before the engine partitions the model, so the head count it records
counts the model's heads. Column-parallel TP then splits attention projections on dim 0, which
is the axis the heads are on: the per-head width survives the split and the count does not.
Nothing catches that on its own, because the stale count usually still divides the shard.
"""

import pytest
import torch

from deepspeed import resolve_per_head_muon_after_sharding


class _Attn(torch.nn.Module):
    """8 heads of 32, as the whole model sees it."""

    def __init__(self, hidden=256, heads=8, head_dim=32):
        super().__init__()
        self.q_proj = torch.nn.Linear(hidden, heads * head_dim, bias=False)
        self.k_proj = torch.nn.Linear(hidden, heads * head_dim, bias=False)
        self.mlp = torch.nn.Linear(hidden, hidden, bias=False)
        for p, num_heads in ((self.q_proj.weight, heads), (self.k_proj.weight, heads), (self.mlp.weight, None)):
            p.muon_num_heads = num_heads
            p.muon_head_dim = p.shape[0] // num_heads if num_heads else None

    def shard(self, tp: int, rows=None, leaves=("q_proj", "k_proj")):
        """Replace the weights with column-parallel shards, as AutoTP's `_tp_partition` does."""
        for leaf in leaves:
            weight = getattr(self, leaf).weight
            keep = rows if rows is not None else weight.shape[0] // tp
            weight.data = weight.data[:keep].clone()


def test_the_head_count_follows_the_shard():
    """tp=2 leaves 4 heads on this rank; the tag has to say 4, not the model's 8."""
    attn = _Attn()
    attn.shard(tp=2)

    resolve_per_head_muon_after_sharding(attn)

    assert attn.q_proj.weight.shape == (128, 256)
    assert attn.q_proj.weight.muon_num_heads == 4
    assert attn.k_proj.weight.muon_num_heads == 4


def test_the_stale_count_would_have_split_heads_in_half():
    """Why this is not caught by the existing shape check: 128 % 8 == 0.

    The divisibility assert in `_per_head_orthogonalize` passes on the stale count, so without
    this pass Newton-Schulz runs on 8 blocks of 16 - half of each head - and says nothing.
    """
    attn = _Attn()
    attn.shard(tp=2)

    rows, stale = attn.q_proj.weight.shape[0], 8
    assert rows % stale == 0, "the stale count divides, which is why it needs correcting rather than asserting"
    assert rows // stale == 16 != 32


def test_a_shard_that_splits_a_head_is_dropped():
    """A shard that does not hold whole heads has no per-head structure to use."""
    attn = _Attn()
    attn.shard(tp=2, rows=144, leaves=("q_proj", ))  # 4.5 heads of 32

    resolve_per_head_muon_after_sharding(attn)

    assert attn.q_proj.weight.muon_num_heads is None
    assert attn.k_proj.weight.muon_num_heads == 8, "one bad shard does not turn the feature off elsewhere"


def test_an_unsharded_model_keeps_its_count():
    attn = _Attn()

    resolve_per_head_muon_after_sharding(attn)

    assert attn.q_proj.weight.muon_num_heads == 8
    assert attn.q_proj.weight.shape[0] == 256


def test_untagged_parameters_are_left_alone():
    attn = _Attn()
    attn.shard(tp=2)

    resolve_per_head_muon_after_sharding(attn)

    assert getattr(attn.mlp.weight, "muon_num_heads", "MISSING") is None


def test_a_model_with_no_tags_at_all_is_a_no_op():
    """Muon without `per_head_muon`, and every non-Muon model: nothing to resolve, no error."""
    model = torch.nn.Linear(8, 8, bias=False)

    resolve_per_head_muon_after_sharding(model)  # must not raise


def test_it_raises_when_the_flag_ends_up_doing_nothing():
    """Same contract as the tagging pass: an opt-in that silently does nothing is the failure."""
    attn = _Attn()
    attn.shard(tp=2, rows=144)

    with pytest.raises(ValueError, match="sharded across head boundaries"):
        resolve_per_head_muon_after_sharding(attn)
