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

Usage
-----
After calling :func:`~deepspeed.sequence.auto_sp.auto_wrap_model_for_sp` to
wrap the ViT attention layers, attach the appropriate fusion adapter to the
vision-language projection layer **before** the first forward pass.  Choose
the adapter that matches your model architecture::

    from deepspeed.sequence.auto_sp import auto_wrap_model_for_sp
    from deepspeed.sequence.autosp_fusion import (
        LlavaFusionAdapter,
        InternVLFusionAdapter,
        Qwen2VLFusionAdapter,
    )
    from deepspeed.utils import groups

    # 1. Wrap ViT and LLM attention layers automatically.
    sp_group = groups._get_sequence_parallel_group()
    auto_wrap_model_for_sp(model, process_group=sp_group)

    # 2. Attach the fusion adapter for the vision-language projection layer.
    #    LLaVA — replaces image-placeholder tokens with visual tokens:
    model.mm_projector = LlavaFusionAdapter(
        model.mm_projector, sp_group, image_token_id=IMAGE_TOKEN_ID
    )

    #    InternVL — replaces IMG_CONTEXT tokens 1-to-1 with visual tokens:
    model.mm_projector = InternVLFusionAdapter(
        model.mm_projector, sp_group, image_token_id=IMG_CONTEXT_TOKEN_ID
    )

    #    Qwen2-VL — replaces tokens between vision_start/end pairs 1-to-1:
    model.visual.merger = Qwen2VLFusionAdapter(
        model.visual.merger, sp_group,
        vision_start_token_id=VISION_START_ID,
        vision_end_token_id=VISION_END_ID,
    )

    # 3. Use the model as normal; the adapter handles all SP gather/scatter.
    outputs = model(input_ids=input_ids, pixel_values=pixel_values, ...)

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
        raise NotImplementedError(f"{type(self).__name__}._splice_visual_into_text is not implemented. "
                                  "Subclass ModalityFusionSPAdapter and override this method to match "
                                  "your model's prepare_inputs_embeds logic.")


class LlavaFusionAdapter(ModalityFusionSPAdapter):
    """LLaVA-style splice: replace each image placeholder token with visual tokens.

    Follows the logic of ``LlavaMetaForCausalLM.prepare_inputs_labels_for_multimodal``:
    for each sample, locate ``image_token_id`` placeholders in ``input_ids``,
    remove them, and insert the corresponding visual token chunk in their place.

    Visual tokens for a sample are split evenly across the number of image
    placeholders found.  This matches the common single-image case (one
    placeholder per sample) and simple multi-image cases where every image
    contributes the same number of tokens.

    Parameters are inherited from :class:`ModalityFusionSPAdapter`.
    """

    def _splice_visual_into_text(self, text_embeds: torch.Tensor, visual_embeds: torch.Tensor,
                                 input_ids: torch.Tensor) -> torch.Tensor:
        bs, text_len, hidden = text_embeds.shape
        device = text_embeds.device

        fused_samples = []
        for i in range(bs):
            img_pos = (input_ids[i] == self.image_token_id).nonzero(as_tuple=True)[0]
            num_images = img_pos.numel()

            if num_images == 0:
                # No image in this sample — keep text embeddings unchanged.
                fused_samples.append(text_embeds[i])
                continue

            # Split all visual tokens evenly across the image placeholders.
            visual_chunks = torch.chunk(visual_embeds[i], num_images, dim=0)

            segments = []
            prev = 0
            for j, pos in enumerate(img_pos.tolist()):
                # Text segment before this placeholder.
                if pos > prev:
                    segments.append(text_embeds[i, prev:pos])
                # Visual tokens replacing this placeholder.
                segments.append(visual_chunks[j])
                # Skip the placeholder token itself.
                prev = pos + 1

            # Remaining text after the last placeholder.
            if prev < text_len:
                segments.append(text_embeds[i, prev:])

            fused_samples.append(torch.cat(segments, dim=0))

        # Pad all samples to the same length so they stack into a tensor.
        max_len = max(s.shape[0] for s in fused_samples)
        out = torch.zeros(bs, max_len, hidden, dtype=text_embeds.dtype, device=device)
        for i, s in enumerate(fused_samples):
            out[i, :s.shape[0]] = s
        return out


class InternVLFusionAdapter(ModalityFusionSPAdapter):
    """InternVL-style splice: replace IMG_CONTEXT token runs with visual tokens.

    InternVL encodes each image as ``<IMG_START> <IMG_CONTEXT>×N <IMG_END>``
    inside the token sequence.  Each ``IMG_CONTEXT`` token (``image_token_id``)
    is a 1-to-1 placeholder for one ViT visual token.  This adapter locates
    every contiguous run of ``image_token_id`` tokens and replaces them with
    the corresponding slice of *visual_embeds*, while preserving the
    ``IMG_START`` / ``IMG_END`` boundary embeddings unchanged.

    Because the replacement is 1-to-1, the output sequence length equals the
    input sequence length (no length change).

    Parameters are inherited from :class:`ModalityFusionSPAdapter`.
    Set ``image_token_id`` to the ``IMG_CONTEXT`` token id used by the model
    (e.g. the id of ``<IMG_CONTEXT>``).
    """

    def _splice_visual_into_text(self, text_embeds: torch.Tensor, visual_embeds: torch.Tensor,
                                 input_ids: torch.Tensor) -> torch.Tensor:
        # Start from a clone of text embeddings; we only overwrite context positions.
        out = text_embeds.clone()
        bs = text_embeds.shape[0]

        for i in range(bs):
            ctx_pos = (input_ids[i] == self.image_token_id).nonzero(as_tuple=True)[0]
            if ctx_pos.numel() == 0:
                continue
            # ctx_pos lists every IMG_CONTEXT index in order.  visual_embeds[i]
            # has exactly ctx_pos.numel() tokens (one per context position).
            out[i, ctx_pos] = visual_embeds[i, :ctx_pos.numel()]

        return out


class Qwen2VLFusionAdapter(nn.Module):
    """Qwen2-VL-style splice: visual tokens enclosed by vision_start/end tokens.

    Qwen2-VL wraps each image's visual tokens with a pair of special boundary
    tokens in ``input_ids``: ``vision_start_token_id`` and
    ``vision_end_token_id``.  The placeholder tokens between each
    (start, end) pair are replaced 1-to-1 by the projected visual token
    embeddings.  The boundary token embeddings are kept unchanged.

    Because the replacement is 1-to-1, the output sequence length equals the
    input sequence length.

    Parameters
    ----------
    projection:
        The vision projection module (e.g. ``visual.merger``).
    process_group:
        The sequence-parallel process group.
    vision_start_token_id:
        Token id of ``<|vision_start|>``.
    vision_end_token_id:
        Token id of ``<|vision_end|>``.
    """

    def __init__(self, projection: nn.Module, process_group, vision_start_token_id: int,
                 vision_end_token_id: int) -> None:
        super().__init__()
        self.projection = projection
        self.process_group = process_group
        self.world_size = dist.get_world_size(process_group)
        self.vision_start_token_id = vision_start_token_id
        self.vision_end_token_id = vision_end_token_id

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
            Token IDs used to locate vision_start/end boundaries.
            Shape: ``[bs, text_seq_len]``.

        Returns
        -------
        Sharded fused embedding for this rank.
        Shape: ``[bs, local_fused_len, lm_hidden]``.
        """
        # 1. Project visual features to the LLM hidden dimension.
        visual_embeds = self.projection(visual_features)  # [bs, local_v, lm_hidden]

        # 2. All-gather visual slices from all SP ranks.
        parts = [torch.zeros_like(visual_embeds) for _ in range(self.world_size)]
        dist.all_gather(parts, visual_embeds.contiguous(), group=self.process_group)
        full_visual = torch.cat(parts, dim=1)  # [bs, total_visual_tokens, lm_hidden]

        # 3. Replace placeholder positions in text with visual tokens (length-preserving).
        fused = self._splice_visual_into_text(text_embeds, full_visual, input_ids)

        # 4. Pad fused length to be divisible by world_size, then scatter.
        total_len = fused.shape[1]
        pad = (self.world_size - total_len % self.world_size) % self.world_size
        if pad > 0:
            fused = F.pad(fused, (0, 0, 0, pad))

        rank = dist.get_rank(self.process_group)
        local_len = fused.shape[1] // self.world_size
        return fused[:, rank * local_len:(rank + 1) * local_len, :].contiguous()

    def _splice_visual_into_text(self, text_embeds: torch.Tensor, visual_embeds: torch.Tensor,
                                 input_ids: torch.Tensor) -> torch.Tensor:
        """Replace inner placeholder tokens between vision_start/end pairs with visual embeddings."""
        out = text_embeds.clone()
        bs = text_embeds.shape[0]

        for i in range(bs):
            start_pos = (input_ids[i] == self.vision_start_token_id).nonzero(as_tuple=True)[0]
            end_pos = (input_ids[i] == self.vision_end_token_id).nonzero(as_tuple=True)[0]

            if start_pos.numel() == 0:
                continue

            # Accumulate inner placeholder positions across all start/end pairs.
            # Inner positions are (start+1) .. (end-1) inclusive, i.e. excluding
            # the boundary tokens themselves.
            inner_positions = []
            for s, e in zip(start_pos.tolist(), end_pos.tolist()):
                inner_positions.extend(range(s + 1, e))

            if not inner_positions:
                continue

            inner_pos = torch.tensor(inner_positions, dtype=torch.long, device=text_embeds.device)
            out[i, inner_pos] = visual_embeds[i, :len(inner_positions)]

        return out
