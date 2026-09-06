# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
"""Why per-head Newton-Schulz survives a column-parallel split and the whole-matrix path does not.

Column-parallel tensor parallelism splits an attention projection on dim 0, which is the axis
per-head Newton-Schulz batches over. Each rank therefore holds whole heads, and orthogonalizing
them is the same computation whether the other ranks' heads are present or not. The whole-matrix
path has no such property: orthogonalizing a block of rows is not the same as taking that block
out of the orthogonalization of all of them. See #8367.
"""

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
