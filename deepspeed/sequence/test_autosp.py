# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
"""
Unit tests for AutoSP multimodal sequence parallelism:
  - autosp_detector: model scanning
  - UlyssesSPViTAttention: ViT SP wrapper
  - auto_wrap_model_for_sp: end-to-end wrapping
"""

import pytest
import torch
import torch.nn as nn

from deepspeed.sequence.autosp_detector import (SPModelInfo, _LLM_ATTN_CLASSNAMES, _VIT_ATTN_CLASSNAMES,
                                                detect_model_sp_info)
from deepspeed.sequence.autosp_vit import UlyssesSPViTAttention
from deepspeed.sequence.auto_sp import _set_module_by_name, auto_wrap_model_for_sp

# ---------------------------------------------------------------------------
# Minimal fake modules that mimic the interface of real attention layers
# without requiring a GPU or a real transformer model.
# ---------------------------------------------------------------------------


class _FakeViTAttn(nn.Module):
    """Identity ViT attention — returns hidden_states unchanged."""

    def forward(self, hidden_states, **kwargs):
        return hidden_states


class _FakeViTAttnTuple(nn.Module):
    """ViT attention that returns a (output, weights) tuple."""

    def forward(self, hidden_states, **kwargs):
        weights = torch.zeros(hidden_states.shape[0], 1, hidden_states.shape[1], hidden_states.shape[1])
        return hidden_states, weights


class _FakeLLMAttn(nn.Module):
    """Identity LLM attention."""

    def forward(self, query, key, value, *args, **kwargs):
        return query


# Register fake class names so the detector recognises them
_VIT_ATTN_CLASSNAMES.add("_FakeViTAttn")
_VIT_ATTN_CLASSNAMES.add("_FakeViTAttnTuple")
_LLM_ATTN_CLASSNAMES.add("_FakeLLMAttn")


class _FakeMultimodalModel(nn.Module):
    """Minimal multimodal model with one ViT and one LLM attention layer."""

    def __init__(self):
        super().__init__()
        self.vision_encoder = nn.ModuleList([_FakeViTAttn()])
        self.mm_projector = nn.Linear(64, 64)
        self.llm = nn.ModuleList([_FakeLLMAttn()])


class _FakeViTOnlyModel(nn.Module):

    def __init__(self, num_layers=3):
        super().__init__()
        self.layers = nn.ModuleList([_FakeViTAttn() for _ in range(num_layers)])


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_mock_process_group(world_size: int, rank: int):
    """Return a mock object that satisfies dist.get_world_size / get_rank."""
    import unittest.mock as mock
    import deepspeed.comm as dist

    pg = mock.MagicMock()
    dist.get_world_size = mock.MagicMock(return_value=world_size)
    dist.get_rank = mock.MagicMock(return_value=rank)

    def _fake_all_gather(tensor_list, tensor, group=None):
        for t in tensor_list:
            t.copy_(tensor)

    dist.all_gather = _fake_all_gather
    return pg


# ---------------------------------------------------------------------------
# autosp_detector tests
# ---------------------------------------------------------------------------


class TestAutospDetector:

    def test_detects_vit_and_llm(self):
        model = _FakeMultimodalModel()
        info = detect_model_sp_info(model)
        assert len(info.vit_attn_modules) == 1
        assert len(info.llm_attn_modules) == 1

    def test_detects_vision_projection(self):
        model = _FakeMultimodalModel()
        info = detect_model_sp_info(model)
        assert info.vision_projection_module is not None
        name, module = info.vision_projection_module
        assert "mm_projector" in name

    def test_detects_multiple_vit_layers(self):
        model = _FakeViTOnlyModel(num_layers=4)
        info = detect_model_sp_info(model)
        assert len(info.vit_attn_modules) == 4
        assert len(info.llm_attn_modules) == 0
        assert info.vision_projection_module is None

    def test_empty_model_returns_empty_info(self):
        model = nn.Sequential(nn.Linear(8, 8))
        info = detect_model_sp_info(model)
        assert isinstance(info, SPModelInfo)
        assert len(info.vit_attn_modules) == 0
        assert len(info.llm_attn_modules) == 0

    def test_only_first_projection_is_recorded(self):
        """Multiple projection-like names → only the outermost is recorded."""

        class _M(nn.Module):

            def __init__(self):
                super().__init__()
                self.mm_projector = nn.Sequential(nn.Linear(8, 8))
                self.mm_projector.visual_projection = nn.Linear(8, 8)

        model = _M()
        info = detect_model_sp_info(model)
        assert info.vision_projection_module is not None
        # Should be the outermost "mm_projector", not the nested one
        name, _ = info.vision_projection_module
        assert name == "mm_projector"


# ---------------------------------------------------------------------------
# UlyssesSPViTAttention tests (CPU, rank-0 simulation via mocks)
# ---------------------------------------------------------------------------


class TestUlyssesSPViTAttention:

    @pytest.mark.parametrize("has_cls_token", [True, False])
    @pytest.mark.parametrize("num_patches,world_size", [
        (16, 4),
        (16, 2),
        (9, 3),
    ])
    def test_output_shape_matches_input(self, has_cls_token, num_patches, world_size):
        """Output shape must equal input shape for any padding scenario."""
        pg = _make_mock_process_group(world_size=world_size, rank=0)
        attn = _FakeViTAttn()
        wrapper = UlyssesSPViTAttention(attn, pg, has_cls_token=has_cls_token)

        local_patches = num_patches // world_size
        seq_len = (1 + local_patches) if has_cls_token else local_patches
        x = torch.randn(2, seq_len, 32)

        out = wrapper(x)
        assert out.shape == x.shape, f"Expected {x.shape}, got {out.shape}"

    def test_tuple_output_unwrapped_correctly(self):
        """Wrappers that return (output, weights) tuples are handled."""
        pg = _make_mock_process_group(world_size=2, rank=0)
        attn = _FakeViTAttnTuple()
        wrapper = UlyssesSPViTAttention(attn, pg, has_cls_token=False)

        x = torch.randn(1, 8, 16)  # 8 patches, 2 ranks → 4 local each
        result = wrapper(x)
        # Should return a tuple: (attention_output, attention_weights)
        assert isinstance(result, tuple)
        assert result[0].shape == x.shape

    def test_identity_attn_preserves_values(self):
        """When attn is identity, output values should match input values."""
        world_size = 2
        pg = _make_mock_process_group(world_size=world_size, rank=0)
        attn = _FakeViTAttn()
        wrapper = UlyssesSPViTAttention(attn, pg, has_cls_token=True)

        # Each rank holds cls + 4 local patches
        x = torch.arange(2 * 5 * 4, dtype=torch.float).reshape(2, 5, 4)
        out = wrapper(x)
        # CLS token should be identical
        assert torch.allclose(out[:, :1, :], x[:, :1, :])
        # Local patch slice should match input patches for identity attn
        assert torch.allclose(out[:, 1:, :], x[:, 1:, :])


# ---------------------------------------------------------------------------
# auto_wrap_model_for_sp tests
# ---------------------------------------------------------------------------


class TestAutoWrapModelForSP:

    def test_vit_layers_replaced(self):
        pg = _make_mock_process_group(world_size=2, rank=0)
        model = _FakeViTOnlyModel(num_layers=2)
        auto_wrap_model_for_sp(model, pg)
        for layer in model.layers:
            assert isinstance(layer, UlyssesSPViTAttention)

    def test_raises_on_unknown_model(self):
        pg = _make_mock_process_group(world_size=2, rank=0)
        model = nn.Sequential(nn.Linear(8, 8))
        with pytest.raises(ValueError, match="no recognisable attention"):
            auto_wrap_model_for_sp(model, pg)

    def test_set_module_by_name_shallow(self):
        model = _FakeViTOnlyModel(num_layers=1)
        new_mod = nn.Linear(4, 4)
        _set_module_by_name(model, "layers.0", new_mod)
        assert model.layers[0] is new_mod

    def test_set_module_by_name_deep(self):
        model = _FakeMultimodalModel()
        new_mod = nn.Identity()
        _set_module_by_name(model, "vision_encoder.0", new_mod)
        assert model.vision_encoder[0] is new_mod
