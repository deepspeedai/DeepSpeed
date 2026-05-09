# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
"""
Automatically detect ViT encoder and LLM decoder attention modules in
multimodal models to guide AutoSP injection.

Extend _VIT_ATTN_CLASSNAMES / _LLM_ATTN_CLASSNAMES when adding support for
new model architectures.
"""

import torch.nn as nn
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

# ---------------------------------------------------------------------------
# Architecture registry
# ---------------------------------------------------------------------------

# Known ViT attention class names (HuggingFace transformers naming)
_VIT_ATTN_CLASSNAMES = {
    "ViTAttention",
    "CLIPAttention",
    "SiglipAttention",
    "InternVisionAttention",
    "Qwen2VLVisionAttention",
    "Idefics2VisionAttention",
    "PaliGemmaVisionAttention",
}

# Whether each known ViT class uses a prepended CLS token.
# CLS is replicated on every rank and is NOT sharded across the sequence.
# Defaults to True for unknown classes (safe fallback).
_VIT_HAS_CLS_TOKEN = {
    "ViTAttention": True,
    "CLIPAttention": True,
    "SiglipAttention": False,
    "InternVisionAttention": False,
    "Qwen2VLVisionAttention": False,
    "Idefics2VisionAttention": False,
    "PaliGemmaVisionAttention": False,
}

# Known LLM decoder attention class names
_LLM_ATTN_CLASSNAMES = {
    "LlamaAttention",
    "MistralAttention",
    "Qwen2Attention",
    "InternLM2Attention",
    "GemmaAttention",
    "Phi3Attention",
    "GPTNeoXAttention",
    "FalconAttention",
    "MptAttention",
}

# Common attribute names that hold the vision-language projection layer
_VISION_PROJ_KEYWORDS = (
    "visual_projection",
    "mm_projector",
    "vision_proj",
    "multi_modal_projector",
    "img_projection",
)

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class SPModelInfo:
    """Holds the detection results for a multimodal model."""

    # (dotted_name, module) pairs for ViT attention layers
    vit_attn_modules: List[Tuple[str, nn.Module]] = field(default_factory=list)
    # (dotted_name, module) pairs for LLM decoder attention layers
    llm_attn_modules: List[Tuple[str, nn.Module]] = field(default_factory=list)
    # (dotted_name, module) for the outermost vision-language projection layer
    vision_projection_module: Optional[Tuple[str, nn.Module]] = None


# ---------------------------------------------------------------------------
# Detection logic
# ---------------------------------------------------------------------------


def detect_model_sp_info(model: nn.Module) -> SPModelInfo:
    """Recursively scan *model* and return an :class:`SPModelInfo`.

    The function identifies:
    * ViT encoder attention layers → wrapped with :class:`UlyssesSPViTAttention`
    * LLM decoder attention layers → wrapped with :class:`DistributedAttention`
    * The vision-language projection layer → wrapped with
      :class:`ModalityFusionSPAdapter` (Phase 2)

    To add support for a new architecture, simply register its attention class
    names in ``_VIT_ATTN_CLASSNAMES`` or ``_LLM_ATTN_CLASSNAMES``.
    """
    info = SPModelInfo()
    for name, module in model.named_modules():
        cls_name = type(module).__name__
        if cls_name in _VIT_ATTN_CLASSNAMES:
            info.vit_attn_modules.append((name, module))
        elif cls_name in _LLM_ATTN_CLASSNAMES:
            info.llm_attn_modules.append((name, module))

        # Record only the first (outermost) match to avoid double-wrapping
        # nested projection modules.
        if info.vision_projection_module is None:
            if any(kw in name for kw in _VISION_PROJ_KEYWORDS):
                info.vision_projection_module = (name, module)

    return info
