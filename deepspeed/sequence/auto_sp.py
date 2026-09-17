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
* a warning for LLM decoder attention layers: HuggingFace-style
  ``hidden_states`` attention is incompatible with
  :class:`~deepspeed.sequence.layer.DistributedAttention`'s Q/K/V interface;
  configure LLM sequence parallelism manually.

The vision-language projection layer (Phase 2) is detected and a warning is
emitted; wrap it manually with
:class:`~deepspeed.sequence.autosp_fusion.ModalityFusionSPAdapter` until
Phase 2 automation is implemented.
"""

import logging

import torch.nn as nn

from deepspeed.sequence.autosp_detector import detect_model_sp_info, _VIT_HAS_CLS_TOKEN
from deepspeed.sequence.autosp_vit import UlyssesSPViTAttention

logger = logging.getLogger(__name__)


def auto_wrap_model_for_sp(model: nn.Module, process_group) -> nn.Module:
    """Inject sequence-parallel wrappers into *model* in-place.

    Scans the model's named modules and replaces recognised attention layers
    with their SP-aware equivalents:

    * ViT attention  → :class:`UlyssesSPViTAttention`
    * LLM attention  → warning only (HuggingFace ``hidden_states`` interface
      is incompatible with :class:`DistributedAttention`'s Q/K/V interface)

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
        raise ValueError("auto_wrap_model_for_sp: no recognisable attention modules found. "
                         "Add the model's attention class name(s) to "
                         "_VIT_ATTN_CLASSNAMES or _LLM_ATTN_CLASSNAMES in "
                         "deepspeed/sequence/autosp_detector.py and retry.")

    # ------------------------------------------------------------------
    # Wrap ViT encoder attention layers
    # ------------------------------------------------------------------
    for name, module in info.vit_attn_modules:
        cls_name = type(module).__name__
        # Look up whether this ViT architecture uses a CLS token; default True
        # (safe fallback) for unknown classes not yet in the registry.
        has_cls = _VIT_HAS_CLS_TOKEN.get(cls_name, True)
        wrapped = UlyssesSPViTAttention(module, process_group, has_cls_token=has_cls)
        _set_module_by_name(model, name, wrapped)
        logger.debug("AutoSP: wrapped ViT attention '%s' with UlyssesSPViTAttention (has_cls_token=%s)", name, has_cls)

    logger.info("AutoSP: wrapped %d ViT attention layer(s).", len(info.vit_attn_modules))

    # ------------------------------------------------------------------
    # LLM decoder attention layers — warn, do not auto-wrap
    # ------------------------------------------------------------------
    # DistributedAttention expects a Megatron-style (query, key, value)
    # interface, but every class in _LLM_ATTN_CLASSNAMES uses the
    # HuggingFace hidden_states interface.  Wrapping them silently would
    # produce incorrect behaviour at the first forward pass.  Emit a
    # per-layer warning so the user can configure SP manually.
    for name, module in info.llm_attn_modules:
        logger.warning(
            "AutoSP: LLM attention '%s' (class %s) uses a HuggingFace hidden_states "
            "interface that is incompatible with DistributedAttention's Q/K/V interface. "
            "Skipping auto-wrap. Configure sequence parallelism for this layer manually.", name,
            type(module).__name__)

    if info.llm_attn_modules:
        logger.info("AutoSP: found %d LLM attention layer(s); skipped wrapping (see warnings above).",
                    len(info.llm_attn_modules))

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
