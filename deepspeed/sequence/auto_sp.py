# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
"""
AutoSP: one-call sequence parallelism for multimodal models.

Usage::

    from deepspeed.sequence.auto_sp import auto_wrap_model_for_sp
    from deepspeed.utils import groups

    model, _, _, _ = deepspeed.initialize(config=ds_config, model=model, ...)
    sp_group = groups._get_sequence_parallel_group()
    model = auto_wrap_model_for_sp(model, process_group=sp_group)

``auto_wrap_model_for_sp`` scans the model and injects:

* :class:`~deepspeed.sequence.autosp_vit.UlyssesSPViTAttention`
  for ViT encoder attention layers.
* :class:`~deepspeed.sequence.layer.DistributedAttention`
  for LLM decoder attention layers (Megatron-style Q/K/V interface).

The vision-language projection layer (Phase 2) is detected and a warning is
emitted; wrap it manually with
:class:`~deepspeed.sequence.autosp_fusion.ModalityFusionSPAdapter` until
Phase 2 automation is implemented.
"""

import logging

import torch.nn as nn

from deepspeed.sequence.autosp_detector import detect_model_sp_info
from deepspeed.sequence.autosp_vit import UlyssesSPViTAttention
from deepspeed.sequence.layer import DistributedAttention

logger = logging.getLogger(__name__)


def auto_wrap_model_for_sp(model: nn.Module, process_group) -> nn.Module:
    """Inject sequence-parallel wrappers into *model* in-place.

    Scans the model's named modules and replaces recognised attention layers
    with their SP-aware equivalents:

    * ViT attention  → :class:`UlyssesSPViTAttention`
    * LLM attention  → :class:`DistributedAttention`

    The function modifies *model* in-place **and** returns it for convenience.

    Parameters
    ----------
    model:
        The multimodal model to wrap.  Must be on the correct device before
        calling this function.
    process_group:
        The sequence-parallel process group (from
        ``groups._get_sequence_parallel_group()``).

    Returns
    -------
    The same *model* object with attention layers replaced.

    Raises
    ------
    ValueError
        If no recognisable attention modules are found.  Register the model's
        attention class names in ``autosp_detector._VIT_ATTN_CLASSNAMES`` or
        ``_LLM_ATTN_CLASSNAMES`` to fix this.
    """
    info = detect_model_sp_info(model)

    if not info.vit_attn_modules and not info.llm_attn_modules:
        raise ValueError(
            "auto_wrap_model_for_sp: no recognisable attention modules found. "
            "Add the model's attention class name(s) to "
            "_VIT_ATTN_CLASSNAMES or _LLM_ATTN_CLASSNAMES in "
            "deepspeed/sequence/autosp_detector.py and retry.")

    # ------------------------------------------------------------------
    # Wrap ViT encoder attention layers
    # ------------------------------------------------------------------
    for name, module in info.vit_attn_modules:
        wrapped = UlyssesSPViTAttention(module, process_group)
        _set_module_by_name(model, name, wrapped)
        logger.debug("AutoSP: wrapped ViT attention '%s' with UlyssesSPViTAttention", name)

    logger.info("AutoSP: wrapped %d ViT attention layer(s).", len(info.vit_attn_modules))

    # ------------------------------------------------------------------
    # Wrap LLM decoder attention layers
    # ------------------------------------------------------------------
    for name, module in info.llm_attn_modules:
        # DistributedAttention wraps a Megatron-style attention that receives
        # (query, key, value) tensors separately.  For HuggingFace-style
        # attention that receives hidden_states, use scatter_idx=2 / gather_idx=0
        # defaults which match the typical [bs, seq, heads, dim] layout.
        wrapped = DistributedAttention(local_attention=module, sequence_process_group=process_group)
        _set_module_by_name(model, name, wrapped)
        logger.debug("AutoSP: wrapped LLM attention '%s' with DistributedAttention", name)

    logger.info("AutoSP: wrapped %d LLM attention layer(s).", len(info.llm_attn_modules))

    # ------------------------------------------------------------------
    # Warn about the vision projection layer (Phase 2)
    # ------------------------------------------------------------------
    if info.vision_projection_module is not None:
        proj_name, _ = info.vision_projection_module
        logger.warning(
            "AutoSP detected vision projection layer '%s'.  "
            "ModalityFusionSPAdapter (Phase 2) is not yet automated.  "
            "Wrap this layer manually with ModalityFusionSPAdapter if you "
            "need correct cross-modal sequence gather/scatter.", proj_name)

    return model


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _set_module_by_name(model: nn.Module, dotted_name: str, new_module: nn.Module) -> None:
    """Replace the submodule at *dotted_name* with *new_module* in-place."""
    parts = dotted_name.split(".")
    parent = model
    for part in parts[:-1]:
        parent = getattr(parent, part)
    setattr(parent, parts[-1], new_module)