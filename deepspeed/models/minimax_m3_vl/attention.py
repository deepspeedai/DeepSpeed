# Copyright (c) DeepSpeed Team.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
"""MiniMax-M3 sparse attention with memory and compute linear in the sequence length, as a drop-in
replacement for the sparse attention layers of transformers' ``minimax_m3_vl`` models.

Why a replacement module: on its eager and SDPA paths, transformers' ``MiniMaxM3VLAttention`` turns
the indexer's block selection into a dense ``[batch, heads, seq, seq]`` additive mask and runs dense
attention under it, after the indexer has built ``[batch, index_heads, seq, seq]`` float32 scores.
On MiniMax-M3 (64 heads) the mask alone is 32 GiB at 16,384 tokens, and time grows with the square
of the length. This module keeps the source module's parameters and attribute names (state-dict
keys and tensor-parallel patterns are unchanged) and only changes the forward:

* the indexer runs ``index_query_chunk`` query rows at a time and gives the same block selection;
* :func:`block_sparse_attention` attends over the selected blocks only, with a backward pass.

The indexer is not trained, as in transformers: its output is integer block ids, so no gradient
reaches it. Where this module does not apply it runs the stock forward unchanged: decoding with a
key-value cache, and calls with an attention mask (padding, or the eager attention implementation).
Training with ``attn_implementation="sdpa"`` and unpadded sequences passes no mask.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from deepspeed.models.minimax_m3_vl.block_sparse_attention import block_sparse_attention
from deepspeed.utils import logger

SUPPORTED_SOURCE_CLASSES = ("MiniMaxM3VLAttention", )

_warned_mask_fallback = False


def _apply_rotary(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    """transformers' ``apply_rotary_pos_emb`` for ``[batch, heads, seq, dim]`` tensors: rotate-half
    rotary embedding on the first ``cos.shape[-1]`` channels, the other channels unchanged."""
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    rotary_dim = cos.shape[-1]
    half = rotary_dim // 2

    def rotate(x):
        x_rot, x_pass = x[..., :rotary_dim], x[..., rotary_dim:]
        rotated = torch.cat((-x_rot[..., half:], x_rot[..., :half]), dim=-1)
        return torch.cat((x_rot * cos + rotated * sin, x_pass), dim=-1)

    return rotate(q), rotate(k)


@torch.no_grad()
def select_blocks(indexer: nn.Module, hidden_states: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor,
                  positions: torch.Tensor, query_chunk: int) -> torch.Tensor:
    """The block selection of transformers' ``MiniMaxM3VLIndexer.forward`` (no cache), computed
    ``query_chunk`` query rows at a time: the float32 scores are ``[batch, index_heads, query_chunk,
    seq]`` instead of ``[batch, index_heads, seq, seq]``. Every step is independent per query row, so
    the result is the same.

    Returns ``[batch, index_heads, seq, topk]`` block ids, ``-1`` for an unused slot.
    """
    batch, seq_len, _ = hidden_states.shape
    head_dim, block_size = indexer.head_dim, indexer.block_size
    index_q = indexer.q_norm(indexer.q_proj(hidden_states).view(batch, seq_len, -1, head_dim)).transpose(1, 2)
    index_k = indexer.k_norm(indexer.k_proj(hidden_states).view(batch, seq_len, 1, head_dim)).transpose(1, 2)
    index_q, index_k = _apply_rotary(index_q, index_k, cos[..., :head_dim], sin[..., :head_dim])

    num_blocks = -(-seq_len // block_size)
    pad = num_blocks * block_size - seq_len
    keys_t = index_k.float().transpose(-1, -2)
    key_positions = torch.arange(seq_len, device=hidden_states.device)
    topk = min(indexer.topk_blocks, num_blocks)
    index_heads = index_q.shape[1]

    selected = []
    for start in range(0, seq_len, query_chunk):
        end = min(start + query_chunk, seq_len)
        chunk_positions = positions[:, start:end]
        scores = torch.matmul(index_q[:, :, start:end].float(), keys_t)
        future = key_positions[None, None, None, :] > chunk_positions[:, None, :, None]
        scores = scores.masked_fill(future, float("-inf"))
        if pad:
            scores = nn.functional.pad(scores, (0, pad), value=float("-inf"))
        block_scores = scores.view(batch, index_heads, end - start, num_blocks, block_size).amax(dim=-1)
        if indexer.local_blocks > 0:
            # The query's own block and the blocks just before it always win a slot.
            local = torch.arange(indexer.local_blocks, device=hidden_states.device)
            local_blocks = (chunk_positions // block_size)[..., None] - local.view(1, 1, -1)
            local_blocks = local_blocks.clamp(min=0).unsqueeze(1).expand(-1, index_heads, -1, -1)
            block_scores.scatter_(-1, local_blocks, float("inf"))
        top_scores, top_blocks = block_scores.topk(topk, dim=-1)
        selected.append(top_blocks.masked_fill(top_scores == float("-inf"), -1))
    return torch.cat(selected, dim=2)


class DeepSpeedMiniMaxM3VLAttention(nn.Module):
    """A MiniMax-M3 sparse attention layer computed with :func:`block_sparse_attention`. See the
    module docstring."""

    def __init__(self, source: nn.Module, query_chunk: int = 512, index_query_chunk: int = 2048):
        super().__init__()
        indexer = source.indexer
        if indexer is None:
            raise ValueError(f"layer {source.layer_idx} has dense attention; only sparse layers can be replaced")
        config = source.config
        if indexer.num_heads != config.num_key_value_heads:
            raise ValueError(f"one indexer head per key-value head is required, got {indexer.num_heads} indexer "
                             f"heads and {config.num_key_value_heads} key-value heads")

        # Every attribute the stock forward reads, so it can run for the calls this module does not cover.
        self.config = config
        self.layer_idx = source.layer_idx
        self.head_dim = source.head_dim
        self.num_key_value_groups = source.num_key_value_groups
        self.scaling = source.scaling
        self.attention_dropout = source.attention_dropout
        self.is_causal = source.is_causal
        self.q_proj = source.q_proj
        self.k_proj = source.k_proj
        self.v_proj = source.v_proj
        self.o_proj = source.o_proj
        self.q_norm = source.q_norm
        self.k_norm = source.k_norm
        self.indexer = indexer
        self._source_class = type(source)

        self.query_chunk = query_chunk
        self.index_query_chunk = index_query_chunk

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None,
        past_key_values=None,
        **kwargs,
    ):
        if past_key_values is not None or attention_mask is not None:
            if attention_mask is not None:
                _warn_mask_fallback_once()
            return self._source_class.forward(self, hidden_states, position_embeddings, attention_mask,
                                              past_key_values, **kwargs)
        if self.training and self.attention_dropout:
            raise NotImplementedError("DeepSpeedMiniMaxM3VLAttention does not implement attention dropout.")

        batch, seq_len, _ = hidden_states.shape
        hidden_shape = (batch, seq_len, -1, self.head_dim)
        query = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        key = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        value = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        cos, sin = position_embeddings
        query, key = _apply_rotary(query, key, cos, sin)

        position_ids = kwargs.get("position_ids")
        if position_ids is None:
            position_ids = torch.arange(seq_len, device=hidden_states.device)
        if position_ids.ndim == 1:
            position_ids = position_ids.unsqueeze(0)
        position_ids = position_ids.expand(batch, -1)

        block_indices = select_blocks(self.indexer, hidden_states, cos, sin, position_ids, self.index_query_chunk)
        out = block_sparse_attention(query, key, value, block_indices, position_ids, self.indexer.block_size,
                                     self.scaling, self.query_chunk)
        return self.o_proj(out.reshape(batch, seq_len, -1)), None


def _warn_mask_fallback_once() -> None:
    global _warned_mask_fallback
    if not _warned_mask_fallback:
        _warned_mask_fallback = True
        logger.warning("DeepSpeedMiniMaxM3VLAttention received an attention mask and runs the stock "
                       "MiniMaxM3VLAttention forward, whose memory grows with the square of the sequence "
                       "length. Train with attn_implementation='sdpa' and unpadded sequences to avoid it.")


def replace_attention(model: nn.Module, query_chunk: int = 512, index_query_chunk: int = 2048) -> int:
    """Replace every sparse attention layer of ``model`` with :class:`DeepSpeedMiniMaxM3VLAttention`.

    Dense attention layers (no indexer) are left as they are. Returns the number of replaced
    modules. Call it after the model is built and before ``deepspeed.initialize``; the parameters are
    reused, not copied, so it also works on a model built under ``deepspeed.zero.Init``.

    ``query_chunk`` query rows go through the attention at a time and ``index_query_chunk`` through
    the indexer; smaller values lower the peak memory and add Python loop steps.
    """
    replaced = 0
    for name, module in list(model.named_modules()):
        if type(module).__name__ not in SUPPORTED_SOURCE_CLASSES or module.indexer is None:
            continue
        parent_name, _, attr = name.rpartition(".")
        parent = model.get_submodule(parent_name) if parent_name else model
        setattr(parent, attr, DeepSpeedMiniMaxM3VLAttention(module, query_chunk, index_query_chunk))
        replaced += 1
    logger.info(f"MiniMax-M3: replaced {replaced} sparse attention module(s) with DeepSpeedMiniMaxM3VLAttention")
    return replaced
