# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
"""CPU-only tests for the TP weight bridges.

These exercise the parallel-kind table and the per-rank slicing math without
requiring vLLM, GPUs, or real model checkpoints.
"""

import pytest
import torch

from opsd.weight_bridge import ParallelKind, Qwen2WeightBridge, Qwen3WeightBridge, get_bridge

# Realistic-ish shapes for a Qwen2.5-0.5B-style model: hidden=896, num_heads=14,
# num_kv_heads=2, head_dim=64, intermediate=4864, vocab=151936. Picked so all
# the per-dim sizes are divisible by tp_size=2.
HIDDEN = 896
NUM_HEADS = 14
NUM_KV_HEADS = 2
HEAD_DIM = 64
INTERMEDIATE = 4864
VOCAB = 151936


def _qwen2_named_tensors():
    """A minimal stand-in for one layer of a Qwen2 state dict."""
    q_dim = NUM_HEADS * HEAD_DIM
    kv_dim = NUM_KV_HEADS * HEAD_DIM
    return [
        ("model.embed_tokens.weight", torch.randn(VOCAB, HIDDEN)),
        ("model.layers.0.self_attn.q_proj.weight", torch.randn(q_dim, HIDDEN)),
        ("model.layers.0.self_attn.k_proj.weight", torch.randn(kv_dim, HIDDEN)),
        ("model.layers.0.self_attn.v_proj.weight", torch.randn(kv_dim, HIDDEN)),
        ("model.layers.0.self_attn.q_proj.bias", torch.randn(q_dim)),
        ("model.layers.0.self_attn.k_proj.bias", torch.randn(kv_dim)),
        ("model.layers.0.self_attn.v_proj.bias", torch.randn(kv_dim)),
        ("model.layers.0.self_attn.o_proj.weight", torch.randn(HIDDEN, q_dim)),
        ("model.layers.0.mlp.gate_proj.weight", torch.randn(INTERMEDIATE, HIDDEN)),
        ("model.layers.0.mlp.up_proj.weight", torch.randn(INTERMEDIATE, HIDDEN)),
        ("model.layers.0.mlp.down_proj.weight", torch.randn(HIDDEN, INTERMEDIATE)),
        ("model.layers.0.input_layernorm.weight", torch.randn(HIDDEN)),
        ("model.layers.0.post_attention_layernorm.weight", torch.randn(HIDDEN)),
        ("model.norm.weight", torch.randn(HIDDEN)),
        ("lm_head.weight", torch.randn(VOCAB, HIDDEN)),
    ]


# --- parallel kind dispatch -------------------------------------------------


@pytest.mark.parametrize("name, expected", [
    ("model.embed_tokens.weight", ParallelKind.VOCAB),
    ("model.layers.0.self_attn.q_proj.weight", ParallelKind.COLUMN),
    ("model.layers.0.self_attn.k_proj.weight", ParallelKind.COLUMN),
    ("model.layers.0.self_attn.v_proj.weight", ParallelKind.COLUMN),
    ("model.layers.42.self_attn.q_proj.bias", ParallelKind.COLUMN),
    ("model.layers.3.self_attn.o_proj.weight", ParallelKind.ROW),
    ("model.layers.3.mlp.gate_proj.weight", ParallelKind.COLUMN),
    ("model.layers.3.mlp.up_proj.weight", ParallelKind.COLUMN),
    ("model.layers.3.mlp.down_proj.weight", ParallelKind.ROW),
    ("model.layers.0.input_layernorm.weight", ParallelKind.REPLICATED),
    ("model.layers.0.post_attention_layernorm.weight", ParallelKind.REPLICATED),
    ("model.norm.weight", ParallelKind.REPLICATED),
    ("lm_head.weight", ParallelKind.VOCAB),
])
def test_qwen2_parallel_kinds(name, expected):
    assert Qwen2WeightBridge().parallel_kind(name) == expected


def test_qwen2_unknown_layer_param_raises():
    with pytest.raises(KeyError, match="Unknown per-layer Qwen2"):
        Qwen2WeightBridge().parallel_kind("model.layers.0.self_attn.q_norm.weight")


def test_qwen2_unknown_global_param_raises():
    with pytest.raises(KeyError, match="Unknown Qwen2 parameter"):
        Qwen2WeightBridge().parallel_kind("totally.made.up.weight")


def test_qwen3_adds_qk_norm():
    bridge = Qwen3WeightBridge()
    assert bridge.parallel_kind("model.layers.0.self_attn.q_norm.weight") == ParallelKind.REPLICATED
    assert bridge.parallel_kind("model.layers.0.self_attn.k_norm.weight") == ParallelKind.REPLICATED
    # Inherits the rest from Qwen2.
    assert bridge.parallel_kind("model.layers.0.self_attn.q_proj.weight") == ParallelKind.COLUMN


# --- slicing math -----------------------------------------------------------


@pytest.mark.parametrize("tp_size", [1, 2, 4])
def test_column_slice_shapes(tp_size):
    bridge = Qwen2WeightBridge()
    w = torch.randn(NUM_HEADS * HEAD_DIM, HIDDEN)
    for rank in range(tp_size):
        sliced = bridge.slice_for_rank("model.layers.0.self_attn.q_proj.weight", w, rank, tp_size)
        assert sliced.shape == (NUM_HEADS * HEAD_DIM // tp_size, HIDDEN)


@pytest.mark.parametrize("tp_size", [1, 2, 4])
def test_row_slice_shapes(tp_size):
    bridge = Qwen2WeightBridge()
    w = torch.randn(HIDDEN, NUM_HEADS * HEAD_DIM)
    for rank in range(tp_size):
        sliced = bridge.slice_for_rank("model.layers.0.self_attn.o_proj.weight", w, rank, tp_size)
        assert sliced.shape == (HIDDEN, NUM_HEADS * HEAD_DIM // tp_size)


def test_replicated_returns_full_tensor():
    bridge = Qwen2WeightBridge()
    w = torch.randn(HIDDEN)
    for rank in range(4):
        sliced = bridge.slice_for_rank("model.layers.0.input_layernorm.weight", w, rank, tp_size=4)
        assert sliced.shape == w.shape
        assert torch.equal(sliced, w)


def test_column_slices_gather_to_original():
    bridge = Qwen2WeightBridge()
    w = torch.randn(NUM_HEADS * HEAD_DIM, HIDDEN)
    tp_size = 2
    pieces = [bridge.slice_for_rank("model.layers.0.self_attn.q_proj.weight", w, r, tp_size) for r in range(tp_size)]
    assert torch.equal(torch.cat(pieces, dim=0), w)


def test_row_slices_gather_to_original():
    bridge = Qwen2WeightBridge()
    w = torch.randn(HIDDEN, INTERMEDIATE)
    tp_size = 4
    pieces = [bridge.slice_for_rank("model.layers.0.mlp.down_proj.weight", w, r, tp_size) for r in range(tp_size)]
    assert torch.equal(torch.cat(pieces, dim=1), w)


def test_vocab_slices_gather_to_original():
    bridge = Qwen2WeightBridge()
    w = torch.randn(VOCAB, HIDDEN)
    tp_size = 4
    pieces = [bridge.slice_for_rank("model.embed_tokens.weight", w, r, tp_size) for r in range(tp_size)]
    assert torch.equal(torch.cat(pieces, dim=0), w)


def test_bias_column_slices_gather_to_original():
    bridge = Qwen2WeightBridge()
    b = torch.randn(NUM_HEADS * HEAD_DIM)
    tp_size = 2
    pieces = [bridge.slice_for_rank("model.layers.0.self_attn.q_proj.bias", b, r, tp_size) for r in range(tp_size)]
    assert torch.equal(torch.cat(pieces, dim=0), b)


def test_indivisible_shape_raises():
    bridge = Qwen2WeightBridge()
    # 7 is not divisible by 2; should fail loudly rather than truncate.
    w = torch.randn(7, HIDDEN)
    with pytest.raises(ValueError, match="not divisible by"):
        bridge.slice_for_rank("model.layers.0.self_attn.q_proj.weight", w, 0, 2)


def test_invalid_rank_raises():
    bridge = Qwen2WeightBridge()
    w = torch.randn(NUM_HEADS * HEAD_DIM, HIDDEN)
    with pytest.raises(ValueError, match="invalid tp_rank"):
        bridge.slice_for_rank("model.layers.0.self_attn.q_proj.weight", w, 4, 4)
    with pytest.raises(ValueError, match="invalid tp_rank"):
        bridge.slice_for_rank("model.layers.0.self_attn.q_proj.weight", w, -1, 2)


def test_row_parallel_rejects_1d():
    """The defensive check inside ``slice_for_rank`` is unreachable through
    the real Qwen2 table (row-parallel biases are tagged REPLICATED), but a
    future bridge could route a 1-D tensor through ROW. Exercise via a
    minimal subclass so the guard stays covered."""

    class _BadBridge(Qwen2WeightBridge):

        def parallel_kind(self, hf_name):  # noqa: ARG002
            return ParallelKind.ROW

    with pytest.raises(ValueError, match="ROW parallel kind requires"):
        _BadBridge().slice_for_rank("anything", torch.randn(HIDDEN), 0, 2)


def test_tp1_is_passthrough():
    bridge = Qwen2WeightBridge()
    w = torch.randn(NUM_HEADS * HEAD_DIM, HIDDEN)
    out = bridge.slice_for_rank("model.layers.0.self_attn.q_proj.weight", w, 0, 1)
    assert torch.equal(out, w)


# --- state-dict iteration ---------------------------------------------------


def test_map_state_dict_emits_correct_shapes_for_tp2():
    bridge = Qwen2WeightBridge()
    tp_size = 2
    # Build the source once; each rank consumes a fresh iterator over the
    # same materialised list so we're slicing identical tensors.
    src = _qwen2_named_tensors()
    by_rank = {r: dict(bridge.map_state_dict(iter(src), r, tp_size)) for r in range(tp_size)}
    src_by_name = dict(src)

    # Replicated tensors should be identical across ranks AND match source.
    a = by_rank[0]["model.layers.0.input_layernorm.weight"]
    b = by_rank[1]["model.layers.0.input_layernorm.weight"]
    assert torch.equal(a, b)
    assert torch.equal(a, src_by_name["model.layers.0.input_layernorm.weight"])

    # Column-parallel Q: shapes halved on dim 0; gather reconstructs source.
    q_full_rows = NUM_HEADS * HEAD_DIM
    assert by_rank[0]["model.layers.0.self_attn.q_proj.weight"].shape == (q_full_rows // 2, HIDDEN)
    gathered_q = torch.cat([
        by_rank[0]["model.layers.0.self_attn.q_proj.weight"],
        by_rank[1]["model.layers.0.self_attn.q_proj.weight"],
    ],
                           dim=0)
    assert torch.equal(gathered_q, src_by_name["model.layers.0.self_attn.q_proj.weight"])


def test_map_state_dict_gather_round_trip_with_fixed_seed():
    bridge = Qwen2WeightBridge()
    torch.manual_seed(123)
    src = _qwen2_named_tensors()
    src_by_name = dict(src)

    tp_size = 4
    sliced = [list(bridge.map_state_dict(src, r, tp_size)) for r in range(tp_size)]

    # For every entry, reconstruct from per-rank slices and compare to the
    # source. The reconstruction op depends on the parallel kind.
    for r0_name, _ in sliced[0]:
        kind = bridge.parallel_kind(r0_name)
        per_rank = [dict(s)[r0_name] for s in sliced]
        if kind is ParallelKind.REPLICATED:
            recon = per_rank[0]
        elif kind in (ParallelKind.COLUMN, ParallelKind.VOCAB):
            recon = torch.cat(per_rank, dim=0)
        elif kind is ParallelKind.ROW:
            recon = torch.cat(per_rank, dim=1)
        else:
            raise AssertionError(f"unhandled kind {kind}")
        assert torch.equal(recon, src_by_name[r0_name]), f"round-trip mismatch for {r0_name}"


# --- registry ---------------------------------------------------------------


def test_get_bridge_qwen2():
    assert isinstance(get_bridge("qwen2"), Qwen2WeightBridge)
    assert isinstance(get_bridge("Qwen2.5"), Qwen2WeightBridge)


def test_get_bridge_qwen3():
    assert isinstance(get_bridge("qwen3"), Qwen3WeightBridge)


def test_get_bridge_unknown_raises():
    with pytest.raises(ValueError, match="No weight bridge registered"):
        get_bridge("totally-made-up-arch")
