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

* the indexer runs ``index_query_chunk`` query rows at a time and gives the same block selection; with
  ``kernel="auto"`` or ``"triton"`` and bf16 or fp16 inputs on an accelerator that supports Triton,
  :func:`index_block_max` instead computes each query's per-block maximum scores in one kernel, without the
  float32 score tensor (exact products and float32 sums, so only near-ties between blocks can order differently);
* the attention reads the selected blocks only, with a backward pass: ``kernel="triton"`` runs
  :func:`selected_attention` (Triton kernels adapted from flash-linear-attention, about 10x faster on H200),
  ``kernel="torch"`` runs :func:`block_sparse_attention` (plain PyTorch), and ``kernel="auto"`` (the default)
  picks Triton for bf16 or fp16 tensors on an accelerator that supports Triton when it takes the shapes
  (``selected_attention_supported``), and PyTorch otherwise.

The indexer is not trained, as in transformers: its output is integer block ids, so no gradient
reaches it. Where this module does not apply it runs the stock forward unchanged: decoding with a
key-value cache, and calls with an attention mask (padding, or the eager attention implementation).
Training with ``attn_implementation="sdpa"`` and unpadded sequences passes no mask.

Context parallel (``set_context_parallel_group``): the ranks of a group hold consecutive equal chunks
of one sequence (batch size 1), rank ``r`` the tokens ``[r * S / cp, (r + 1) * S / cp)``, and the
caller passes the matching ``position_ids``. Queries stay local. The keys and values (4 key-value
heads) and the indexer's single key head are gathered over the group, so every query selects and
reads its blocks with global positions. The gradient of the gathered keys and values is
reduce-scattered back to the owning ranks. The dense attention layers need the same treatment, so
``replace_attention(..., dense=True)`` also replaces them with :class:`DeepSpeedMiniMaxM3VLDenseAttention`.
Everything outside attention works per token, so the rest of the model needs no change.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from deepspeed.accelerator import get_accelerator
from deepspeed.models.minimax_m3_vl.block_sparse_attention import block_sparse_attention
from deepspeed.models.minimax_m3_vl.index_kernel import index_block_max
from deepspeed.models.minimax_m3_vl.selected_attention import selected_attention, selected_attention_supported
from deepspeed.ops.triton_ops import is_triton_available
from deepspeed.utils import logger

SUPPORTED_SOURCE_CLASSES = ("MiniMaxM3VLAttention", )

_warned_mask_fallback = False
_CONTEXT_PARALLEL_GROUP = None


def set_context_parallel_group(group) -> None:
    """Split each sequence over the ranks of ``group`` (``None`` turns the split off). See the module docstring."""
    global _CONTEXT_PARALLEL_GROUP
    _CONTEXT_PARALLEL_GROUP = group


def _context_parallel():
    """``(group, size, rank)`` of the context-parallel group, or ``(None, 1, 0)``."""
    import deepspeed.comm as dist
    group = _CONTEXT_PARALLEL_GROUP
    if group is None:
        return None, 1, 0
    return group, dist.get_world_size(group=group), dist.get_rank(group=group)


def _all_gather_tokens(local: torch.Tensor, group, size: int) -> torch.Tensor:
    import deepspeed.comm as dist
    local = local.contiguous()
    full = torch.empty((local.shape[0] * size, *local.shape[1:]), dtype=local.dtype, device=local.device)
    dist.all_gather_into_tensor(full, local, group=group)
    return full


class _GatherTokens(torch.autograd.Function):
    """All-gather along dim 0 over the context-parallel group; the backward reduce-scatters (sums) the gradient."""

    @staticmethod
    def forward(ctx, local: torch.Tensor, group, size: int) -> torch.Tensor:
        ctx.group, ctx.size = group, size
        return _all_gather_tokens(local, group, size)

    @staticmethod
    def backward(ctx, grad_full: torch.Tensor):
        import deepspeed.comm as dist
        grad_full = grad_full.contiguous()
        grad_local = torch.empty((grad_full.shape[0] // ctx.size, *grad_full.shape[1:]),
                                 dtype=grad_full.dtype,
                                 device=grad_full.device)
        dist.reduce_scatter_tensor(grad_local, grad_full, op=dist.ReduceOp.SUM, group=ctx.group)
        return grad_local, None, None


def _gather_sequence(x: torch.Tensor, group, size: int) -> torch.Tensor:
    """``[1, heads, local_seq, dim]`` to ``[1, heads, local_seq * size, dim]``: the whole sequence of the group."""
    if size == 1:
        return x
    if x.shape[0] != 1:
        raise ValueError(f"context parallel needs batch size 1, got {x.shape[0]}")
    tokens_first = x[0].transpose(0, 1)  # [local_seq, heads, dim]
    if torch.is_grad_enabled() and x.requires_grad:
        full = _GatherTokens.apply(tokens_first, group, size)
    else:
        full = _all_gather_tokens(tokens_first, group, size)
    return full.transpose(0, 1).unsqueeze(0)


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
def _index_kernel_supported(q: torch.Tensor, keys: torch.Tensor, block_size: int) -> bool:
    head_dim = q.shape[-1]
    if q.device.type == "cpu" or q.dtype not in (torch.bfloat16, torch.float16) or keys.dtype != q.dtype:
        return False
    if head_dim & (head_dim - 1) or block_size & (block_size - 1):
        return False
    return is_triton_available() and get_accelerator().is_triton_supported()


def select_blocks(indexer: nn.Module,
                  hidden_states: torch.Tensor,
                  cos: torch.Tensor,
                  sin: torch.Tensor,
                  positions: torch.Tensor,
                  query_chunk: int,
                  use_kernel: bool = False) -> torch.Tensor:
    """The block selection of transformers' ``MiniMaxM3VLIndexer.forward`` (no cache), computed
    ``query_chunk`` query rows at a time: the float32 scores are ``[batch, index_heads, query_chunk,
    seq]`` instead of ``[batch, index_heads, seq, seq]``. Every step is independent per query row, so
    the result is the same.

    ``use_kernel``: compute the per-block maximum scores of all local queries with :func:`index_block_max`
    when the inputs allow it (bf16 or fp16 on an accelerator with Triton), instead of the float32 matmul.

    Returns ``[batch, index_heads, seq, topk]`` block ids, ``-1`` for an unused slot.
    """
    batch, seq_len, _ = hidden_states.shape
    head_dim, block_size = indexer.head_dim, indexer.block_size
    index_q = indexer.q_norm(indexer.q_proj(hidden_states).view(batch, seq_len, -1, head_dim)).transpose(1, 2)
    index_k = indexer.k_norm(indexer.k_proj(hidden_states).view(batch, seq_len, 1, head_dim)).transpose(1, 2)
    index_q, index_k = _apply_rotary(index_q, index_k, cos[..., :head_dim], sin[..., :head_dim])
    # Under context parallel the local queries score the keys of the whole sequence.
    group, size, _ = _context_parallel()
    index_k = _gather_sequence(index_k, group, size)
    key_len = index_k.shape[2]

    num_blocks = -(-key_len // block_size)
    pad = num_blocks * block_size - key_len
    key_positions = torch.arange(key_len, device=hidden_states.device)
    topk = min(indexer.topk_blocks, num_blocks)
    index_heads = index_q.shape[1]
    use_kernel = use_kernel and _index_kernel_supported(index_q, index_k, block_size)
    if use_kernel:
        # The kernel's output is only [batch, index_heads, seq, num_blocks], so all local queries go at once.
        query_chunk = seq_len
    else:
        keys_t = index_k.float().transpose(-1, -2)

    selected = []
    for start in range(0, seq_len, query_chunk):
        end = min(start + query_chunk, seq_len)
        chunk_positions = positions[:, start:end]
        if use_kernel:
            block_scores = index_block_max(index_q[:, :, start:end], index_k, chunk_positions, block_size)
        else:
            # Float32 as in the model code, also under an engine's bf16 autocast (which would run this matmul, and so
            # the mask and the block maxima, in bf16).
            with torch.autocast(device_type=index_q.device.type, enabled=False):
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
    """A MiniMax-M3 sparse attention layer computed with :func:`selected_attention` or
    :func:`block_sparse_attention` (``kernel``). See the module docstring."""

    def __init__(self, source: nn.Module, query_chunk: int = 512, index_query_chunk: int = 2048, kernel: str = "auto"):
        super().__init__()
        if kernel not in ("auto", "triton", "torch"):
            raise ValueError(f"kernel must be 'auto', 'triton' or 'torch', got {kernel!r}")
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
        self.kernel = kernel

    def _use_selected_attention(self, query: torch.Tensor, kv_heads: int) -> bool:
        if self.kernel != "auto":
            return self.kernel == "triton"
        # The Triton kernels compute in half precision on an accelerator that runs Triton; float32 and CPU
        # tensors take the PyTorch path, which keeps float32 exact.
        if query.device.type == "cpu" or query.dtype not in (torch.bfloat16, torch.float16):
            return False
        if not (is_triton_available() and get_accelerator().is_triton_supported()):
            return False
        return selected_attention_supported(query.shape[1], kv_heads, self.head_dim, self.indexer.block_size)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None,
        past_key_values=None,
        **kwargs,
    ):
        group, size, rank = _context_parallel()
        if past_key_values is not None or attention_mask is not None:
            if size > 1:
                raise ValueError("context parallel supports neither a key-value cache nor an attention mask")
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
            if size > 1:
                raise ValueError("context parallel needs the global position_ids of the local tokens")
            position_ids = torch.arange(seq_len, device=hidden_states.device)
        if position_ids.ndim == 1:
            position_ids = position_ids.unsqueeze(0)
        position_ids = position_ids.expand(batch, -1)

        block_indices = select_blocks(self.indexer,
                                      hidden_states,
                                      cos,
                                      sin,
                                      position_ids,
                                      self.index_query_chunk,
                                      use_kernel=self.kernel != "torch")
        key = _gather_sequence(key, group, size)
        value = _gather_sequence(value, group, size)
        if self._use_selected_attention(query, key.shape[1]):
            # The kernel takes the queries as the last positions of the keys it is given: rank r of the split holds
            # queries [r * seq_len, (r + 1) * seq_len), so it gets keys [0, (r + 1) * seq_len); without the split,
            # position_ids must be 0 .. seq_len - 1.
            visible = (rank + 1) * seq_len
            out = selected_attention(query.transpose(1, 2),
                                     key[:, :, :visible].transpose(1, 2), value[:, :, :visible].transpose(1, 2),
                                     block_indices.permute(0, 2, 1, 3), self.indexer.block_size, self.scaling)
        else:
            out = block_sparse_attention(query, key, value, block_indices, position_ids, self.indexer.block_size,
                                         self.scaling, self.query_chunk)
        return self.o_proj(out.reshape(batch, seq_len, -1)), None


class DeepSpeedMiniMaxM3VLDenseAttention(nn.Module):
    """A MiniMax-M3 dense (full causal) attention layer that also works under context parallel: the local
    queries attend to the gathered keys and values of every earlier token of the sequence. Without
    context parallel it is the stock SDPA computation."""

    def __init__(self, source: nn.Module):
        super().__init__()
        if source.indexer is not None:
            raise ValueError(f"layer {source.layer_idx} has sparse attention; use DeepSpeedMiniMaxM3VLAttention")
        self.config = source.config
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
        self.indexer = None
        self._source_class = type(source)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None,
        past_key_values=None,
        **kwargs,
    ):
        group, size, rank = _context_parallel()
        if past_key_values is not None or attention_mask is not None:
            if size > 1:
                raise ValueError("context parallel supports neither a key-value cache nor an attention mask")
            if attention_mask is not None:
                _warn_mask_fallback_once()
            return self._source_class.forward(self, hidden_states, position_embeddings, attention_mask,
                                              past_key_values, **kwargs)
        if self.training and self.attention_dropout:
            raise NotImplementedError("DeepSpeedMiniMaxM3VLDenseAttention does not implement attention dropout.")

        batch, seq_len, _ = hidden_states.shape
        hidden_shape = (batch, seq_len, -1, self.head_dim)
        query = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        key = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        value = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        cos, sin = position_embeddings
        query, key = _apply_rotary(query, key, cos, sin)

        if size == 1:
            mask, is_causal = None, True
        else:
            # Rank r's queries are tokens [r * seq_len, (r + 1) * seq_len); they see the keys before their own
            # end, which is causal aligned to the lower-right corner of the [seq_len, (r + 1) * seq_len] scores.
            from torch.nn.attention.bias import causal_lower_right
            visible = (rank + 1) * seq_len
            key = _gather_sequence(key, group, size)[:, :, :visible]
            value = _gather_sequence(value, group, size)[:, :, :visible]
            mask, is_causal = causal_lower_right(seq_len, visible), False
        key = key.repeat_interleave(self.num_key_value_groups, dim=1)
        value = value.repeat_interleave(self.num_key_value_groups, dim=1)
        out = nn.functional.scaled_dot_product_attention(query,
                                                         key,
                                                         value,
                                                         attn_mask=mask,
                                                         is_causal=is_causal,
                                                         scale=self.scaling)
        out = out.transpose(1, 2).reshape(batch, seq_len, -1)
        return self.o_proj(out), None


def _warn_mask_fallback_once() -> None:
    global _warned_mask_fallback
    if not _warned_mask_fallback:
        _warned_mask_fallback = True
        logger.warning("DeepSpeedMiniMaxM3VLAttention received an attention mask and runs the stock "
                       "MiniMaxM3VLAttention forward, whose memory grows with the square of the sequence "
                       "length. Train with attn_implementation='sdpa' and unpadded sequences to avoid it.")


def replace_attention(model: nn.Module,
                      query_chunk: int = 512,
                      index_query_chunk: int = 2048,
                      dense: bool = False,
                      kernel: str = "auto") -> int:
    """Replace every sparse attention layer of ``model`` with :class:`DeepSpeedMiniMaxM3VLAttention`, and with
    ``dense=True`` every dense layer with :class:`DeepSpeedMiniMaxM3VLDenseAttention` (needed for context parallel).

    Otherwise dense attention layers (no indexer) are left as they are. Returns the number of replaced
    modules. Call it after the model is built and before ``deepspeed.initialize``; the parameters are
    reused, not copied, so it also works on a model built under ``deepspeed.zero.Init``.

    ``query_chunk`` query rows go through the PyTorch attention at a time and ``index_query_chunk``
    through the indexer; smaller values lower the peak memory and add Python loop steps. ``kernel``
    chooses the sparse attention: ``"auto"``, ``"triton"`` or ``"torch"`` (see the module docstring).
    """
    replaced = 0
    for name, module in list(model.named_modules()):
        if type(module).__name__ not in SUPPORTED_SOURCE_CLASSES:
            continue
        if module.indexer is not None:
            replacement = DeepSpeedMiniMaxM3VLAttention(module, query_chunk, index_query_chunk, kernel)
        elif dense:
            replacement = DeepSpeedMiniMaxM3VLDenseAttention(module)
        else:
            continue
        parent_name, _, attr = name.rpartition(".")
        parent = model.get_submodule(parent_name) if parent_name else model
        setattr(parent, attr, replacement)
        replaced += 1
    logger.info(f"MiniMax-M3: replaced {replaced} attention module(s) (dense layers included: {dense}, "
                f"sparse attention kernel: {kernel})")
    return replaced
