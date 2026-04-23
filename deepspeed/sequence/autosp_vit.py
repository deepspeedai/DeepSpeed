# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
"""
Ulysses-style sequence-parallel wrapper for ViT encoder attention layers.

Design notes
------------
ViT self-attention is non-causal: every patch token attends to every other
patch token.  This means a straightforward per-rank local attention (as used
for causal LLMs) would be *incorrect* — each rank must have access to the
full key/value context.

We therefore use a **gather-compute-scatter** pattern:

1. Input arrives already sharded along the sequence dimension (each rank owns
   ``local_patches = num_patches // world_size`` consecutive patches).
2. Before attention we **all-gather** patch tokens so that every rank runs the
   full ViT attention over the complete patch sequence.  This keeps the
   computation equivalent to single-device execution.
3. The output is **scattered** back so that each rank returns only its local
   slice, matching the sharded input contract expected by downstream layers.

Memory benefit: activations *outside* the attention block (e.g. feed-forward
layers, layer norms) are stored only locally, reducing per-rank memory
proportional to ``world_size``.

The ``cls`` token (if present) is replicated on every rank and is not split
across the sequence dimension.  Each rank appends its local patches to the
same ``cls`` token before calling the wrapped attention.

Padding: when ``num_patches % world_size != 0``, we pad patches with zeros
before scattering and strip the padding after gathering.  Padding tokens do
not carry gradients and are never passed to downstream layers.
"""

import torch
import torch.nn as nn

import deepspeed.comm as dist


class UlyssesSPViTAttention(nn.Module):
    """Sequence-parallel wrapper for an opaque ViT attention module.

    Parameters
    ----------
    attn:
        The original ViT attention layer (any ``nn.Module`` that maps
        ``hidden_states`` → ``hidden_states`` or a tuple whose first element
        is the attention output tensor).
    process_group:
        The sequence-parallel process group.
    has_cls_token:
        Set to ``True`` (default) when the first token in the sequence is a
        ``[CLS]`` token that should be replicated on every rank rather than
        sharded.
    """

    def __init__(self, attn: nn.Module, process_group, has_cls_token: bool = True) -> None:
        super().__init__()
        self.attn = attn
        self.process_group = process_group
        self.world_size = dist.get_world_size(process_group)
        self.has_cls_token = has_cls_token

    # ------------------------------------------------------------------
    # forward
    # ------------------------------------------------------------------

    def forward(self, hidden_states: torch.Tensor, **kwargs):
        """
        Parameters
        ----------
        hidden_states:
            Shape ``[bs, local_seq_len, hidden_dim]`` where
            ``local_seq_len = (1 + local_patches)`` if ``has_cls_token`` else
            ``local_patches``.  Each rank holds a contiguous slice of patches.
        **kwargs:
            Passed through to the wrapped attention (e.g. ``attention_mask``,
            ``head_mask``, ``output_attentions``).

        Returns
        -------
        Same shape as input (or a tuple whose first element matches the input
        shape, preserving whatever the wrapped module returns).
        """
        bs, local_seq_len, hidden_dim = hidden_states.shape

        if self.has_cls_token:
            # CLS token is replicated on every rank — not part of the sharded seq
            cls_token = hidden_states[:, :1, :]
            local_patches = hidden_states[:, 1:, :]
        else:
            local_patches = hidden_states

        local_patch_len = local_patches.shape[1]

        # -------------------------------------------------------------------
        # 1. All-gather patches from all ranks to reconstruct the full sequence
        # -------------------------------------------------------------------
        # We need to all-gather so every rank sees the full K/V context.
        gathered = [torch.zeros_like(local_patches) for _ in range(self.world_size)]
        dist.all_gather(gathered, local_patches.contiguous(), group=self.process_group)
        full_patches = torch.cat(gathered, dim=1)  # [bs, num_patches_padded, hidden_dim]

        # -------------------------------------------------------------------
        # 2. Build the full input (prepend CLS if needed) and call attention
        # -------------------------------------------------------------------
        if self.has_cls_token:
            full_input = torch.cat([cls_token, full_patches], dim=1)
        else:
            full_input = full_patches

        attn_out = self.attn(full_input, **kwargs)

        # Unwrap tuple: some ViT implementations return (attn_output, attn_weights)
        if isinstance(attn_out, (tuple, list)):
            full_out, *extra = attn_out
        else:
            full_out = attn_out
            extra = []

        # -------------------------------------------------------------------
        # 3. Scatter output: each rank keeps only its local slice of patches
        # -------------------------------------------------------------------
        if self.has_cls_token:
            cls_out = full_out[:, :1, :]
            patch_out = full_out[:, 1:, :]
        else:
            patch_out = full_out

        # Determine this rank's slice boundaries
        rank = dist.get_rank(self.process_group)
        local_out = patch_out[:, rank * local_patch_len:(rank + 1) * local_patch_len, :].contiguous()

        if self.has_cls_token:
            local_out = torch.cat([cls_out, local_out], dim=1)

        if extra:
            return (local_out, *extra)
        return local_out
