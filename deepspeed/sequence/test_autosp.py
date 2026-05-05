# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
"""
Unit tests for AutoSP multimodal sequence parallelism:
  - autosp_detector: model scanning
  - UlyssesSPViTAttention: ViT SP wrapper
  - auto_wrap_model_for_sp: end-to-end wrapping
  - ModalityFusionSPAdapter: cross-modal gather/scatter
  - LlavaFusionAdapter: LLaVA-style visual token splice
  - InternVLFusionAdapter: InternVL-style IMG_CONTEXT token splice
  - Qwen2VLFusionAdapter: Qwen2-VL vision_start/end bounded splice
"""

import pytest
import torch
import torch.nn as nn

from deepspeed.sequence.autosp_detector import (SPModelInfo, _LLM_ATTN_CLASSNAMES, _VIT_ATTN_CLASSNAMES,
                                                detect_model_sp_info)
from deepspeed.sequence.autosp_fusion import (InternVLFusionAdapter, LlavaFusionAdapter, ModalityFusionSPAdapter,
                                              Qwen2VLFusionAdapter)
from deepspeed.sequence.autosp_vit import UlyssesSPViTAttention
from deepspeed.sequence.auto_sp import _set_module_by_name, auto_wrap_model_for_sp
from deepspeed.sequence.layer import DistributedAttention

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


class _FakeLLMOnlyModel(nn.Module):
    """Minimal LLM-only model with multiple decoder attention layers."""

    def __init__(self, num_layers=2):
        super().__init__()
        self.layers = nn.ModuleList([_FakeLLMAttn() for _ in range(num_layers)])


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

    def test_llm_layers_replaced_with_distributed_attention(self):
        """LLM attention layers must be wrapped with DistributedAttention."""
        pg = _make_mock_process_group(world_size=2, rank=0)
        model = _FakeLLMOnlyModel(num_layers=3)
        auto_wrap_model_for_sp(model, pg)
        for layer in model.layers:
            assert isinstance(layer, DistributedAttention)

    def test_multimodal_model_wraps_both_branches(self):
        """Both ViT and LLM attention layers must be replaced in a combined model."""
        pg = _make_mock_process_group(world_size=2, rank=0)
        model = _FakeMultimodalModel()
        returned = auto_wrap_model_for_sp(model, pg)
        # auto_wrap_model_for_sp must return the same object (in-place)
        assert returned is model
        assert isinstance(model.vision_encoder[0], UlyssesSPViTAttention)
        assert isinstance(model.llm[0], DistributedAttention)

    def test_original_module_preserved_inside_wrapper(self):
        """The wrapped module should still be accessible inside the wrapper."""
        pg = _make_mock_process_group(world_size=2, rank=0)
        model = _FakeViTOnlyModel(num_layers=1)
        original_attn = model.layers[0]
        auto_wrap_model_for_sp(model, pg)
        assert model.layers[0].attn is original_attn


# ---------------------------------------------------------------------------
# ModalityFusionSPAdapter tests
# ---------------------------------------------------------------------------


class _ConcatFusionAdapter(ModalityFusionSPAdapter):
    """Concrete subclass that appends visual tokens after text tokens."""

    def _splice_visual_into_text(self, text_embeds, visual_embeds, input_ids):
        return torch.cat([text_embeds, visual_embeds], dim=1)


class TestModalityFusionSPAdapter:

    def test_base_class_raises_not_implemented(self):
        """The base _splice_visual_into_text must raise NotImplementedError."""
        pg = _make_mock_process_group(world_size=2, rank=0)
        adapter = ModalityFusionSPAdapter(nn.Identity(), pg)
        with pytest.raises(NotImplementedError):
            adapter._splice_visual_into_text(None, None, None)

    @pytest.mark.parametrize("world_size,local_v,text_len,hidden", [
        (2, 4, 6, 8),
        (4, 3, 5, 16),
        (1, 8, 8, 4),
    ])
    def test_output_shape(self, world_size, local_v, text_len, hidden):
        """Output local_len must equal ceil(fused_len / world_size)."""
        pg = _make_mock_process_group(world_size=world_size, rank=0)
        adapter = _ConcatFusionAdapter(nn.Identity(), pg)

        bs = 2
        visual = torch.randn(bs, local_v, hidden)
        text = torch.randn(bs, text_len, hidden)
        ids = torch.zeros(bs, text_len, dtype=torch.long)

        out = adapter(visual, text, ids)

        # all_gather mock copies local_v to each of world_size slots
        fused_len = text_len + local_v * world_size
        pad = (world_size - fused_len % world_size) % world_size
        expected_local = (fused_len + pad) // world_size
        assert out.shape == (bs, expected_local, hidden), f"Expected ({bs},{expected_local},{hidden}), got {out.shape}"

    def test_padding_produces_valid_output_when_not_divisible(self):
        """When fused_len % world_size != 0, padding must not raise and output is well-formed."""
        world_size = 4
        # text_len=5, local_v=3 → fused_len = 5 + 3*4 = 17, needs padding of 3
        pg = _make_mock_process_group(world_size=world_size, rank=0)
        adapter = _ConcatFusionAdapter(nn.Identity(), pg)

        bs, local_v, text_len, hidden = 1, 3, 5, 4
        out = adapter(
            torch.randn(bs, local_v, hidden),
            torch.randn(bs, text_len, hidden),
            torch.zeros(bs, text_len, dtype=torch.long),
        )
        # padded_len = 20, local_len = 5
        assert out.shape == (bs, 5, hidden)

    def test_no_padding_when_divisible(self):
        """When fused_len is already divisible, no extra tokens should be added."""
        world_size = 4
        # text_len=4, local_v=4 → fused_len = 4 + 4*4 = 20, divisible by 4
        pg = _make_mock_process_group(world_size=world_size, rank=0)
        adapter = _ConcatFusionAdapter(nn.Identity(), pg)

        bs, local_v, text_len, hidden = 1, 4, 4, 8
        out = adapter(
            torch.randn(bs, local_v, hidden),
            torch.randn(bs, text_len, hidden),
            torch.zeros(bs, text_len, dtype=torch.long),
        )
        assert out.shape == (bs, 5, hidden)  # 20 // 4 = 5

    def test_different_ranks_return_different_slices(self):
        """Rank 0 and rank 1 must return different slices of the fused sequence."""
        world_size = 2
        bs, local_v, text_len, hidden = 1, 4, 4, 8
        # Use distinct text vs visual values so slices clearly differ
        text = torch.zeros(bs, text_len, hidden)
        visual = torch.ones(bs, local_v, hidden)
        ids = torch.zeros(bs, text_len, dtype=torch.long)

        outputs = {}
        for rank in range(world_size):
            pg = _make_mock_process_group(world_size=world_size, rank=rank)
            adapter = _ConcatFusionAdapter(nn.Identity(), pg)
            outputs[rank] = adapter(visual.clone(), text.clone(), ids.clone())

        # fused = [0,0,0,0, 1,1,1,1, 1,1,1,1]  (text zeros then visual ones x2)
        # rank 0: indices 0-5, rank 1: indices 6-11
        assert not torch.allclose(outputs[0], outputs[1])

    def test_projection_is_applied(self):
        """Projection layer must transform visual features before gather."""
        world_size = 2
        pg = _make_mock_process_group(world_size=world_size, rank=0)

        # Use a projection that doubles all values
        class _DoubleProjection(nn.Module):

            def forward(self, x):
                return x * 2.0

        adapter = _ConcatFusionAdapter(_DoubleProjection(), pg)
        bs, local_v, text_len, hidden = 1, 4, 4, 8
        visual = torch.ones(bs, local_v, hidden)
        text = torch.zeros(bs, text_len, hidden)
        ids = torch.zeros(bs, text_len, dtype=torch.long)

        out = adapter(visual, text, ids)
        # The visual part of the output should have value 2.0 (doubled), not 1.0
        # rank 0 gets the first local_len tokens; fused = [text(0)*4, visual(2)*8]
        # Since text_len=4 and local_len=6, rank0 slice starts with text zeros
        # and ends with some visual twos.
        assert out.max().item() == pytest.approx(2.0)


# ---------------------------------------------------------------------------
# LlavaFusionAdapter tests  (tests _splice_visual_into_text directly)
# ---------------------------------------------------------------------------

_IMAGE_ID = -200  # matches ModalityFusionSPAdapter default


def _make_llava_adapter(world_size=2, rank=0):
    pg = _make_mock_process_group(world_size=world_size, rank=rank)
    return LlavaFusionAdapter(nn.Identity(), pg, image_token_id=_IMAGE_ID)


class TestLlavaFusionAdapter:

    def test_single_image_fused_shape(self):
        """One image placeholder per sample → fused length = text_len - 1 + num_visual."""
        adapter = _make_llava_adapter()
        bs, text_len, num_vis, hidden = 2, 6, 4, 8
        # Place a single image placeholder at position 2.
        ids = torch.zeros(bs, text_len, dtype=torch.long)
        ids[:, 2] = _IMAGE_ID
        text = torch.randn(bs, text_len, hidden)
        visual = torch.randn(bs, num_vis, hidden)

        fused = adapter._splice_visual_into_text(text, visual, ids)
        # placeholder is removed and replaced by num_vis tokens
        assert fused.shape == (bs, text_len - 1 + num_vis, hidden)

    def test_text_values_preserved_around_image(self):
        """Text tokens before and after the placeholder must be numerically intact."""
        adapter = _make_llava_adapter()
        bs, text_len, num_vis, hidden = 1, 5, 3, 4
        # Placeholder at index 2: text = [A, B, <img>, C, D]
        ids = torch.zeros(bs, text_len, dtype=torch.long)
        ids[0, 2] = _IMAGE_ID
        text = torch.arange(bs * text_len * hidden, dtype=torch.float).reshape(bs, text_len, hidden)
        visual = torch.ones(bs, num_vis, hidden) * 99.0

        fused = adapter._splice_visual_into_text(text, visual, ids)
        # fused = [A, B, vis0, vis1, vis2, C, D]
        assert torch.allclose(fused[0, :2], text[0, :2])  # A, B preserved
        assert torch.allclose(fused[0, 5:], text[0, 3:])  # C, D preserved
        assert torch.allclose(fused[0, 2:5], visual[0])  # visual inserted

    def test_no_image_token_returns_text_unchanged(self):
        """When input_ids has no placeholder, output equals text_embeds exactly."""
        adapter = _make_llava_adapter()
        bs, text_len, hidden = 2, 6, 8
        ids = torch.zeros(bs, text_len, dtype=torch.long)  # no -200
        text = torch.randn(bs, text_len, hidden)
        visual = torch.randn(bs, 4, hidden)

        fused = adapter._splice_visual_into_text(text, visual, ids)
        assert fused.shape == (bs, text_len, hidden)
        assert torch.allclose(fused, text)

    def test_multi_image_splice(self):
        """Two placeholders per sample → visual tokens split evenly between them."""
        adapter = _make_llava_adapter()
        bs, text_len, num_vis, hidden = 1, 7, 6, 4
        # Placeholders at index 1 and 4: [t0, <img>, t2, t3, <img>, t5, t6]
        ids = torch.zeros(bs, text_len, dtype=torch.long)
        ids[0, 1] = _IMAGE_ID
        ids[0, 4] = _IMAGE_ID
        text = torch.zeros(bs, text_len, hidden)
        # First 3 visual tokens = 1.0, last 3 = 2.0 (so we can tell them apart)
        visual = torch.cat([torch.ones(bs, 3, hidden), torch.full((bs, 3, hidden), 2.0)], dim=1)

        fused = adapter._splice_visual_into_text(text, visual, ids)
        # Expected fused length: 7 - 2 placeholders + 6 visual = 11
        assert fused.shape == (bs, 11, hidden)
        # First chunk (indices 1-3) should be 1.0
        assert torch.allclose(fused[0, 1:4], torch.ones(3, hidden))
        # Second chunk (indices 6-8) should be 2.0
        assert torch.allclose(fused[0, 6:9], torch.full((3, hidden), 2.0))

    def test_batch_padding_when_lengths_differ(self):
        """Samples with different numbers of image tokens are padded to max length."""
        adapter = _make_llava_adapter()
        hidden = 4
        # Sample 0: 1 placeholder in a 4-token sequence + 2 visual → fused len = 5
        # Sample 1: no placeholder in a 4-token sequence → fused len = 4
        ids = torch.zeros(2, 4, dtype=torch.long)
        ids[0, 1] = _IMAGE_ID
        text = torch.ones(2, 4, hidden)
        visual = torch.ones(2, 2, hidden) * 3.0

        fused = adapter._splice_visual_into_text(text, visual, ids)
        # Max fused length is 5; sample 1 padded with zeros at the end.
        assert fused.shape == (2, 5, hidden)
        assert torch.all(fused[1, 4:] == 0)  # padding tokens are zero

    def test_forward_end_to_end_shape(self):
        """Full forward pass through LlavaFusionAdapter returns the correct shard shape."""
        world_size = 2
        pg = _make_mock_process_group(world_size=world_size, rank=0)
        adapter = LlavaFusionAdapter(nn.Identity(), pg, image_token_id=_IMAGE_ID)

        bs, local_v, text_len, hidden = 1, 4, 6, 8
        ids = torch.zeros(bs, text_len, dtype=torch.long)
        ids[0, 2] = _IMAGE_ID  # one placeholder
        visual = torch.randn(bs, local_v, hidden)
        text = torch.randn(bs, text_len, hidden)

        out = adapter(visual, text, ids)
        # fused_len = text_len - 1 + local_v * world_size = 5 + 8 = 13
        # padded to 14 (next multiple of 2), local = 7
        assert out.shape == (bs, 7, hidden)


# ---------------------------------------------------------------------------
# InternVLFusionAdapter tests  (tests _splice_visual_into_text directly)
# ---------------------------------------------------------------------------

_CONTEXT_ID = 92546  # arbitrary IMG_CONTEXT token id for tests
_START_ID = 92545
_END_ID = 92547


def _make_internvl_adapter(world_size=2, rank=0):
    pg = _make_mock_process_group(world_size=world_size, rank=rank)
    return InternVLFusionAdapter(nn.Identity(), pg, image_token_id=_CONTEXT_ID)


class TestInternVLFusionAdapter:

    def test_context_tokens_replaced_with_visual(self):
        """IMG_CONTEXT positions must carry visual embeddings after splice."""
        adapter = _make_internvl_adapter()
        bs, text_len, hidden = 1, 7, 4
        # Layout: [t0, START, ctx, ctx, ctx, END, t6]
        ids = torch.zeros(bs, text_len, dtype=torch.long)
        ids[0, 2] = _CONTEXT_ID
        ids[0, 3] = _CONTEXT_ID
        ids[0, 4] = _CONTEXT_ID

        text = torch.zeros(bs, text_len, hidden)
        visual = torch.ones(bs, 3, hidden) * 7.0

        fused = adapter._splice_visual_into_text(text, visual, ids)
        assert torch.allclose(fused[0, 2:5], visual[0])

    def test_sequence_length_preserved(self):
        """Output length must equal input length (1-to-1 replacement)."""
        adapter = _make_internvl_adapter()
        bs, text_len, hidden = 2, 10, 8
        ids = torch.zeros(bs, text_len, dtype=torch.long)
        ids[:, 3:7] = _CONTEXT_ID  # 4 context tokens per sample
        text = torch.randn(bs, text_len, hidden)
        visual = torch.randn(bs, 4, hidden)

        fused = adapter._splice_visual_into_text(text, visual, ids)
        assert fused.shape == (bs, text_len, hidden)

    def test_boundary_tokens_preserved(self):
        """IMG_START and IMG_END embeddings must be unchanged after splice."""
        adapter = _make_internvl_adapter()
        bs, text_len, hidden = 1, 5, 4
        # [START, ctx, ctx, END, text]
        ids = torch.zeros(bs, text_len, dtype=torch.long)
        ids[0, 1] = _CONTEXT_ID
        ids[0, 2] = _CONTEXT_ID

        text = torch.arange(bs * text_len * hidden, dtype=torch.float).reshape(bs, text_len, hidden)
        visual = torch.ones(bs, 2, hidden) * 99.0

        fused = adapter._splice_visual_into_text(text, visual, ids)
        # Position 0 (START) and 3 (END) must be unchanged.
        assert torch.allclose(fused[0, 0], text[0, 0])
        assert torch.allclose(fused[0, 3], text[0, 3])

    def test_no_context_tokens_returns_text_unchanged(self):
        """When there are no IMG_CONTEXT tokens the output must equal text_embeds."""
        adapter = _make_internvl_adapter()
        bs, text_len, hidden = 2, 6, 8
        ids = torch.zeros(bs, text_len, dtype=torch.long)
        text = torch.randn(bs, text_len, hidden)
        visual = torch.randn(bs, 4, hidden)

        fused = adapter._splice_visual_into_text(text, visual, ids)
        assert torch.allclose(fused, text)

    def test_multi_image_replacement(self):
        """Two separate runs of context tokens correspond to two images."""
        adapter = _make_internvl_adapter()
        bs, text_len, hidden = 1, 10, 4
        # Image 1: positions 1-2, Image 2: positions 6-7
        ids = torch.zeros(bs, text_len, dtype=torch.long)
        ids[0, 1] = _CONTEXT_ID
        ids[0, 2] = _CONTEXT_ID
        ids[0, 6] = _CONTEXT_ID
        ids[0, 7] = _CONTEXT_ID

        text = torch.zeros(bs, text_len, hidden)
        # First 2 visual tokens = 1.0, next 2 = 2.0
        visual = torch.cat([torch.ones(bs, 2, hidden), torch.full((bs, 2, hidden), 2.0)], dim=1)

        fused = adapter._splice_visual_into_text(text, visual, ids)
        assert fused.shape == (bs, text_len, hidden)
        assert torch.allclose(fused[0, 1:3], torch.ones(2, hidden))
        assert torch.allclose(fused[0, 6:8], torch.full((2, hidden), 2.0))

    def test_forward_end_to_end_shape(self):
        """Full forward pass returns the correct shard shape."""
        world_size = 2
        pg = _make_mock_process_group(world_size=world_size, rank=0)
        adapter = InternVLFusionAdapter(nn.Identity(), pg, image_token_id=_CONTEXT_ID)

        bs, local_v, text_len, hidden = 1, 3, 8, 4
        ids = torch.zeros(bs, text_len, dtype=torch.long)
        ids[0, 2:5] = _CONTEXT_ID  # 3 context tokens; local_v * world_size = 6 total
        visual = torch.randn(bs, local_v, hidden)
        text = torch.randn(bs, text_len, hidden)

        out = adapter(visual, text, ids)
        # fused_len == text_len == 8 (length-preserving); padded to 8 (divisible by 2); local = 4
        assert out.shape == (bs, 4, hidden)


# ---------------------------------------------------------------------------
# Qwen2VLFusionAdapter tests  (tests _splice_visual_into_text directly)
# ---------------------------------------------------------------------------

_VIS_START_ID = 151652
_VIS_END_ID = 151653


def _make_qwen2vl_adapter(world_size=2, rank=0):
    pg = _make_mock_process_group(world_size=world_size, rank=rank)
    return Qwen2VLFusionAdapter(nn.Identity(),
                                pg,
                                vision_start_token_id=_VIS_START_ID,
                                vision_end_token_id=_VIS_END_ID)


class TestQwen2VLFusionAdapter:

    def test_inner_tokens_replaced_with_visual(self):
        """Tokens between vision_start and vision_end must become visual embeddings."""
        adapter = _make_qwen2vl_adapter()
        bs, text_len, hidden = 1, 7, 4
        # [t0, t1, <vis_start>, pad, pad, <vis_end>, t6]
        ids = torch.zeros(bs, text_len, dtype=torch.long)
        ids[0, 2] = _VIS_START_ID
        ids[0, 5] = _VIS_END_ID

        text = torch.zeros(bs, text_len, hidden)
        visual = torch.ones(bs, 2, hidden) * 5.0

        fused = adapter._splice_visual_into_text(text, visual, ids)
        assert torch.allclose(fused[0, 3:5], visual[0])

    def test_sequence_length_preserved(self):
        """Output length must equal input length (1-to-1 replacement)."""
        adapter = _make_qwen2vl_adapter()
        bs, text_len, hidden = 2, 12, 8
        ids = torch.zeros(bs, text_len, dtype=torch.long)
        ids[:, 2] = _VIS_START_ID
        ids[:, 8] = _VIS_END_ID  # 5 inner placeholder tokens
        text = torch.randn(bs, text_len, hidden)
        visual = torch.randn(bs, 5, hidden)

        fused = adapter._splice_visual_into_text(text, visual, ids)
        assert fused.shape == (bs, text_len, hidden)

    def test_boundary_tokens_preserved(self):
        """vision_start and vision_end embeddings must be unchanged after splice."""
        adapter = _make_qwen2vl_adapter()
        bs, text_len, hidden = 1, 6, 4
        # [t0, <vis_start>, pad, pad, <vis_end>, t5]
        ids = torch.zeros(bs, text_len, dtype=torch.long)
        ids[0, 1] = _VIS_START_ID
        ids[0, 4] = _VIS_END_ID

        text = torch.arange(bs * text_len * hidden, dtype=torch.float).reshape(bs, text_len, hidden)
        visual = torch.ones(bs, 2, hidden) * 99.0

        fused = adapter._splice_visual_into_text(text, visual, ids)
        assert torch.allclose(fused[0, 1], text[0, 1])  # vision_start preserved
        assert torch.allclose(fused[0, 4], text[0, 4])  # vision_end preserved

    def test_no_vision_tokens_returns_text_unchanged(self):
        """When there are no vision_start/end tokens the output must equal text_embeds."""
        adapter = _make_qwen2vl_adapter()
        bs, text_len, hidden = 2, 8, 4
        ids = torch.zeros(bs, text_len, dtype=torch.long)
        text = torch.randn(bs, text_len, hidden)
        visual = torch.randn(bs, 4, hidden)

        fused = adapter._splice_visual_into_text(text, visual, ids)
        assert torch.allclose(fused, text)

    def test_multi_image_replacement(self):
        """Two vision blocks are handled independently."""
        adapter = _make_qwen2vl_adapter()
        bs, text_len, hidden = 1, 14, 4
        # Block 1: positions 1 (start) .. 4 (end), 2 inner tokens at 2-3
        # Block 2: positions 8 (start) .. 12 (end), 3 inner tokens at 9-11
        ids = torch.zeros(bs, text_len, dtype=torch.long)
        ids[0, 1] = _VIS_START_ID
        ids[0, 4] = _VIS_END_ID
        ids[0, 8] = _VIS_START_ID
        ids[0, 12] = _VIS_END_ID

        text = torch.zeros(bs, text_len, hidden)
        visual = torch.cat([torch.ones(bs, 2, hidden), torch.full((bs, 3, hidden), 2.0)], dim=1)

        fused = adapter._splice_visual_into_text(text, visual, ids)
        assert fused.shape == (bs, text_len, hidden)
        assert torch.allclose(fused[0, 2:4], torch.ones(2, hidden))
        assert torch.allclose(fused[0, 9:12], torch.full((3, hidden), 2.0))

    def test_forward_end_to_end_shape(self):
        """Full forward pass returns the correct shard shape."""
        world_size = 2
        pg = _make_mock_process_group(world_size=world_size, rank=0)
        adapter = Qwen2VLFusionAdapter(nn.Identity(),
                                       pg,
                                       vision_start_token_id=_VIS_START_ID,
                                       vision_end_token_id=_VIS_END_ID)

        bs, local_v, text_len, hidden = 1, 3, 10, 4
        ids = torch.zeros(bs, text_len, dtype=torch.long)
        # 6 inner placeholder tokens (local_v * world_size = 6)
        ids[0, 1] = _VIS_START_ID
        ids[0, 8] = _VIS_END_ID
        visual = torch.randn(bs, local_v, hidden)
        text = torch.randn(bs, text_len, hidden)

        out = adapter(visual, text, ids)
        # fused_len == text_len == 10 (length-preserving); padded to 10; local = 5
        assert out.shape == (bs, 5, hidden)
