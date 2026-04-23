# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
"""
ModalityFusionSPAdapter — Phase 2

Handles the sequence scatter/gather at the vision-language boundary so that
the LLM decoder's :class:`~deepspeed.sequence.layer.DistributedAttention`
receives a uniformly sharded fused (visual + text) sequence.

Workflow
--------
::

    [visual tokens, sharded]  ──all-gather──►  [visual tokens, full]
                                                        │
                                                 splice into text
                                                        │
    [fused embeds, full]  ──scatter──►  [fused embeds, sharded per rank]
                                                        │
                                               LLM decoder (SP-aware)

Status: Phase 2.  ``_splice_visual_into_text`` is intentionally left as a
``NotImplementedError``; override it per model architecture (see docstring).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

import deepspeed.comm as dist

# Default image placeholder token ID used by LLaVA-style models.
_DEFAULT_IMAGE_TOKEN_ID = -200


class ModalityFusionSPAdapter(nn.Module):
    """Wraps the vision projection layer and handles cross-modal sequence fusion.

    After projecting visual features, this adapter:

    1. Gathers the sharded visual token slices from all SP ranks into a single
       full visual token tensor.
    2. Splices the visual tokens into the text embedding sequence at the
       positions marked by ``image_token_id`` placeholders.
    3. Pads and re-shards the fused sequence so that the subsequent LLM
       decoder layers receive uniformly distributed sequence slices.

    Parameters
    ----------
    projection:
        The vision projection module (e.g. ``mm_projector``).
    process_group:
        The sequence-parallel process group.
    image_token_id:
        The token ID used as an image placeholder in the input IDs tensor.
        Defaults to ``-200`` (LLaVA convention).

    Notes
    -----
    Subclass this and override :meth:`_splice_visual_into_text` to adapt to a
    specific multimodal architecture (LLaVA, InternVL, Qwen-VL, …).
    """

    def __init__(self, projection: nn.Module, process_group, image_token_id: int = _DEFAULT_IMAGE_TOKEN_ID) -> None:
        super().__init__()
        self.projection = projection
        self.process_group = process_group
        self.world_size = dist.get_world_size(process_group)
        self.image_token_id = image_token_id

    def forward(self, visual_features: torch.Tensor, text_embeds: torch.Tensor,
                input_ids: torch.Tensor) -> torch.Tensor:
        """Project visual features and return a sharded fused embedding.

        Parameters
        ----------
        visual_features:
            Raw visual features from the ViT encoder.
            Shape: ``[bs, local_visual_tokens, vit_hidden]``.
        text_embeds:
            Full text token embeddings (not sharded yet).
            Shape: ``[bs, text_seq_len, lm_hidden]``.
        input_ids:
            Token IDs used to locate image placeholder positions.
            Shape: ``[bs, text_seq_len]``.

        Returns
        -------
        Sharded fused embedding for this rank.
        Shape: ``[bs, local_fused_len, lm_hidden]``.
        """
        # 1. Project visual features to the LLM hidden dimension
        visual_embeds = self.projection(visual_features)  # [bs, local_v, lm_hidden]

        # 2. All-gather visual slices from all SP ranks
        parts = [torch.zeros_like(visual_embeds) for _ in range(self.world_size)]
        dist.all_gather(parts, visual_embeds.contiguous(), group=self.process_group)
        full_visual = torch.cat(parts, dim=1)  # [bs, total_visual_tokens, lm_hidden]

        # 3. Splice visual tokens into text embedding sequence
        fused = self._splice_visual_into_text(text_embeds, full_visual, input_ids)  # [bs, fused_len, lm_hidden]

        # 4. Pad fused length to be divisible by world_size, then scatter
        total_len = fused.shape[1]
        pad = (self.world_size - total_len % self.world_size) % self.world_size
        if pad > 0:
            fused = F.pad(fused, (0, 0, 0, pad))

        rank = dist.get_rank(self.process_group)
        local_len = fused.shape[1] // self.world_size
        return fused[:, rank * local_len:(rank + 1) * local_len, :].contiguous()

    def _splice_visual_into_text(self, text_embeds: torch.Tensor, visual_embeds: torch.Tensor,
                                  input_ids: torch.Tensor) -> torch.Tensor:
        """Replace image placeholder positions in *text_embeds* with *visual_embeds*.

        This is intentionally architecture-specific.  The default raises
        ``NotImplementedError``; override this method for each supported model.

        Reference implementations:
        * LLaVA: ``LlavaMetaForCausalLM.prepare_inputs_embeds``
        * InternVL: ``InternVLChatModel.extract_feature``
        * Qwen-VL: ``Qwen2VLForConditionalGeneration.get_rope_index``
        """
        raise NotImplementedError(
            f"{type(self).__name__}._splice_visual_into_text is not implemented. "
            "Subclass ModalityFusionSPAdapter and override this method to match "
            "your model's prepare_inputs_embeds logic.")