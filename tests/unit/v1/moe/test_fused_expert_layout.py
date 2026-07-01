# Copyright (c) DeepSpeed Team.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

from deepspeed.moe.fused_expert_layout import classify_fused_gate_up_layout


def test_classifies_gate_up_first():
    layout = classify_fused_gate_up_layout((8, 256, 64), (8, 64, 128))

    assert layout is not None
    assert layout.layout == "gate_up_first"
    assert layout.hidden_size == 64
    assert layout.ffn_hidden_size == 128
    assert layout.needs_transpose is False


def test_classifies_hidden_first():
    layout = classify_fused_gate_up_layout((8, 64, 256), (8, 128, 64))

    assert layout is not None
    assert layout.layout == "hidden_first"
    assert layout.hidden_size == 64
    assert layout.ffn_hidden_size == 128
    assert layout.needs_transpose is True


def test_returns_none_for_unknown_layout():
    assert classify_fused_gate_up_layout((8, 64, 64), (8, 64, 64)) is None


def test_returns_none_for_odd_inner_dim():
    assert classify_fused_gate_up_layout((8, 255, 64), (8, 64, 127)) is None


def test_returns_none_for_non_3d():
    assert classify_fused_gate_up_layout((64, 64), (64, 64)) is None
