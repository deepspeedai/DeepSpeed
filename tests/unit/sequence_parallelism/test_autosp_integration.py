# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team
"""
End-to-end integration tests for AutoSP multimodal sequence parallelism.

Each test builds a minimal mock model whose attention-layer class names match
the autosp_detector registry, then verifies two things:

1. auto_wrap_model_for_sp correctly identifies and wraps ViT attention modules
   (with the correct has_cls_token value from the registry) and emits warnings
   for HF-style LLM attention without wrapping them.
2. The full pipeline (SP-wrapped ViT -> fusion adapter) produces fused output
   numerically equivalent to the single-device splice reference.

These tests require 2 GPUs.
Run with:

    NCCL_P2P_DISABLE=1 python -m pytest tests/unit/sequence_parallelism/test_autosp_integration.py -v
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

import deepspeed.comm as dist
from deepspeed.sequence.auto_sp import auto_wrap_model_for_sp
from deepspeed.sequence.autosp_vit import UlyssesSPViTAttention
from deepspeed.sequence.autosp_fusion import InternVLFusionAdapter, Qwen2VLFusionAdapter
from deepspeed.accelerator import get_accelerator

from unit.common import DistributedTest

# ---------------------------------------------------------------------------
# Token IDs
# ---------------------------------------------------------------------------

_INTERNVL_CONTEXT_ID = 92546
_QWEN2VL_START_ID = 151652
_QWEN2VL_END_ID = 151653

# ---------------------------------------------------------------------------
# Mock attention classes
#
# Class names must match exactly the entries in autosp_detector._VIT_ATTN_CLASSNAMES
# and _LLM_ATTN_CLASSNAMES so that auto_wrap_model_for_sp detects them.
# ---------------------------------------------------------------------------


class InternVisionAttention(nn.Module):
    """Mock ViT attention for InternVL (registered in _VIT_ATTN_CLASSNAMES)."""

    def forward(self, hidden_states, **kwargs):
        return hidden_states


class InternLM2Attention(nn.Module):
    """Mock LLM attention for InternVL (registered in _LLM_ATTN_CLASSNAMES)."""

    def forward(self, hidden_states, **kwargs):
        return hidden_states


class Qwen2VLVisionAttention(nn.Module):
    """Mock ViT attention for Qwen2-VL (registered in _VIT_ATTN_CLASSNAMES)."""

    def forward(self, hidden_states, **kwargs):
        return hidden_states


class Qwen2Attention(nn.Module):
    """Mock LLM attention for Qwen2-VL (registered in _LLM_ATTN_CLASSNAMES)."""

    def forward(self, hidden_states, **kwargs):
        return hidden_states


# ---------------------------------------------------------------------------
# Model skeleton helpers
# ---------------------------------------------------------------------------


class _AttnLayer(nn.Module):
    """Generic transformer block that holds an attention submodule.

    auto_wrap_model_for_sp scans named_modules() and replaces ``self.attn``
    when its class name is in the detector's registry.
    """

    def __init__(self, attn: nn.Module) -> None:
        super().__init__()
        self.attn = attn

    def forward(self, x, **kwargs):
        return self.attn(x, **kwargs)


class _MinimalInternVLModel(nn.Module):
    """Minimal InternVL-like skeleton for integration testing.

    Module paths recognised by autosp_detector:
    - ``vision_encoder.0.attn``  -> InternVisionAttention  (_VIT_ATTN_CLASSNAMES)
    - ``language_model.0.attn``  -> InternLM2Attention     (_LLM_ATTN_CLASSNAMES)
    - ``mm_projector``           -> keyword in _VISION_PROJ_KEYWORDS

    ``forward`` exercises only the ViT + fusion path; ``language_model`` is
    present to verify that auto_wrap does NOT wrap HF-style LLM attention.
    """

    def __init__(self) -> None:
        super().__init__()
        self.vision_encoder = nn.Sequential(_AttnLayer(InternVisionAttention()))
        self.mm_projector = nn.Identity()
        self.language_model = nn.Sequential(_AttnLayer(InternLM2Attention()))
        self.fusion = None

    def forward(self, local_patches: torch.Tensor, text_embeds: torch.Tensor, input_ids: torch.Tensor) -> torch.Tensor:
        local_visual = self.vision_encoder(local_patches)
        return self.fusion(local_visual, text_embeds, input_ids)


class _MinimalQwen2VLModel(nn.Module):
    """Minimal Qwen2-VL-like skeleton for integration testing.

    Module paths recognised by autosp_detector:
    - ``visual.0.attn``          -> Qwen2VLVisionAttention (_VIT_ATTN_CLASSNAMES)
    - ``model.0.attn``           -> Qwen2Attention          (_LLM_ATTN_CLASSNAMES)
    - ``multi_modal_projector``  -> keyword in _VISION_PROJ_KEYWORDS
    """

    def __init__(self) -> None:
        super().__init__()
        self.visual = nn.Sequential(_AttnLayer(Qwen2VLVisionAttention()))
        self.multi_modal_projector = nn.Identity()
        self.model = nn.Sequential(_AttnLayer(Qwen2Attention()))
        self.fusion = None

    def forward(self, local_patches: torch.Tensor, text_embeds: torch.Tensor, input_ids: torch.Tensor) -> torch.Tensor:
        local_visual = self.visual(local_patches)
        return self.fusion(local_visual, text_embeds, input_ids)


# ---------------------------------------------------------------------------
# InternVL integration tests
# ---------------------------------------------------------------------------


class TestInternVLIntegration(DistributedTest):
    """Integration tests for the InternVL multimodal SP pipeline."""

    world_size = 2

    def test_auto_wrap_detects_and_wraps_modules(self):
        """auto_wrap_model_for_sp must replace InternVisionAttention with
        UlyssesSPViTAttention (has_cls_token=False) and must NOT wrap
        InternLM2Attention (HF-style, incompatible with DistributedAttention)."""
        sp_group = dist.new_group(ranks=list(range(self.world_size)))
        model = _MinimalInternVLModel().to(get_accelerator().device_name())
        auto_wrap_model_for_sp(model, sp_group)

        assert isinstance(
            model.vision_encoder[0].attn,
            UlyssesSPViTAttention), ("Expected vision_encoder[0].attn to be UlyssesSPViTAttention after auto_wrap")
        assert not model.vision_encoder[0].attn.has_cls_token, (
            "InternVisionAttention has no CLS token; has_cls_token must be False")
        assert isinstance(model.language_model[0].attn,
                          InternLM2Attention), ("HF-style LLM attention must NOT be wrapped by auto_wrap")

    def test_full_pipeline_visual_to_fused(self):
        """SP-wrapped ViT -> InternVLFusionAdapter must produce fused output
        numerically equivalent to the single-device splice reference."""
        sp_group = dist.new_group(ranks=list(range(self.world_size)))
        rank = dist.get_rank(sp_group)

        bs, local_v, text_len, hidden = 1, 4, 10, 8
        num_ctx = local_v * self.world_size

        torch.manual_seed(20)
        full_visual = torch.randn(bs, local_v * self.world_size, hidden).to(get_accelerator().device_name())
        text = torch.randn(bs, text_len, hidden).to(get_accelerator().device_name())
        ids = torch.zeros(bs, text_len, dtype=torch.long).to(get_accelerator().device_name())
        ids[:, 2:2 + num_ctx] = _INTERNVL_CONTEXT_ID

        local_patches = full_visual[:, rank * local_v:(rank + 1) * local_v, :]

        model = _MinimalInternVLModel().to(get_accelerator().device_name())
        auto_wrap_model_for_sp(model, sp_group)
        model.fusion = InternVLFusionAdapter(model.mm_projector, sp_group,
                                             image_token_id=_INTERNVL_CONTEXT_ID).to(get_accelerator().device_name())

        local_out = model(local_patches, text, ids)

        gathered = [torch.zeros_like(local_out) for _ in range(self.world_size)]
        dist.all_gather(gathered, local_out, group=sp_group)
        full_sp_out = torch.cat(gathered, dim=1)

        # Single-device reference: splice without SP scatter.
        ref_adapter = InternVLFusionAdapter(nn.Identity(), sp_group,
                                            image_token_id=_INTERNVL_CONTEXT_ID).to(get_accelerator().device_name())
        ref_fused = ref_adapter._splice_visual_into_text(text, full_visual, ids)
        pad = (self.world_size - ref_fused.shape[1] % self.world_size) % self.world_size
        if pad > 0:
            ref_fused = F.pad(ref_fused, (0, 0, 0, pad))

        assert torch.allclose(full_sp_out, ref_fused,
                              atol=1e-5), (f"rank={rank} InternVL full pipeline output differs from reference: "
                                           f"max_diff={(full_sp_out - ref_fused).abs().max().item():.2e}")


# ---------------------------------------------------------------------------
# Qwen2-VL integration tests
# ---------------------------------------------------------------------------


class TestQwen2VLIntegration(DistributedTest):
    """Integration tests for the Qwen2-VL multimodal SP pipeline."""

    world_size = 2

    def test_auto_wrap_detects_and_wraps_modules(self):
        """auto_wrap_model_for_sp must replace Qwen2VLVisionAttention with
        UlyssesSPViTAttention (has_cls_token=False) and must NOT wrap
        Qwen2Attention (HF-style, incompatible with DistributedAttention)."""
        sp_group = dist.new_group(ranks=list(range(self.world_size)))
        model = _MinimalQwen2VLModel().to(get_accelerator().device_name())
        auto_wrap_model_for_sp(model, sp_group)

        assert isinstance(
            model.visual[0].attn,
            UlyssesSPViTAttention), ("Expected visual[0].attn to be UlyssesSPViTAttention after auto_wrap")
        assert not model.visual[0].attn.has_cls_token, (
            "Qwen2VLVisionAttention has no CLS token; has_cls_token must be False")
        assert isinstance(model.model[0].attn,
                          Qwen2Attention), ("HF-style LLM attention must NOT be wrapped by auto_wrap")

    def test_full_pipeline_visual_to_fused(self):
        """SP-wrapped ViT -> Qwen2VLFusionAdapter must produce fused output
        numerically equivalent to the single-device splice reference."""
        sp_group = dist.new_group(ranks=list(range(self.world_size)))
        rank = dist.get_rank(sp_group)

        bs, local_v, text_len, hidden = 1, 3, 10, 8
        num_inner = local_v * self.world_size

        torch.manual_seed(21)
        full_visual = torch.randn(bs, local_v * self.world_size, hidden).to(get_accelerator().device_name())
        text = torch.randn(bs, text_len, hidden).to(get_accelerator().device_name())
        ids = torch.zeros(bs, text_len, dtype=torch.long).to(get_accelerator().device_name())
        ids[:, 1] = _QWEN2VL_START_ID
        ids[:, 2 + num_inner] = _QWEN2VL_END_ID

        local_patches = full_visual[:, rank * local_v:(rank + 1) * local_v, :]

        model = _MinimalQwen2VLModel().to(get_accelerator().device_name())
        auto_wrap_model_for_sp(model, sp_group)
        model.fusion = Qwen2VLFusionAdapter(model.multi_modal_projector,
                                            sp_group,
                                            vision_start_token_id=_QWEN2VL_START_ID,
                                            vision_end_token_id=_QWEN2VL_END_ID).to(get_accelerator().device_name())

        local_out = model(local_patches, text, ids)

        gathered = [torch.zeros_like(local_out) for _ in range(self.world_size)]
        dist.all_gather(gathered, local_out, group=sp_group)
        full_sp_out = torch.cat(gathered, dim=1)

        ref_adapter = Qwen2VLFusionAdapter(nn.Identity(),
                                           sp_group,
                                           vision_start_token_id=_QWEN2VL_START_ID,
                                           vision_end_token_id=_QWEN2VL_END_ID).to(get_accelerator().device_name())
        ref_fused = ref_adapter._splice_visual_into_text(text, full_visual, ids)
        pad = (self.world_size - ref_fused.shape[1] % self.world_size) % self.world_size
        if pad > 0:
            ref_fused = F.pad(ref_fused, (0, 0, 0, pad))

        assert torch.allclose(full_sp_out, ref_fused,
                              atol=1e-5), (f"rank={rank} Qwen2VL full pipeline output differs from reference: "
                                           f"max_diff={(full_sp_out - ref_fused).abs().max().item():.2e}")
