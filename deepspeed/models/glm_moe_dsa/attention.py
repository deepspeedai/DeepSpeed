# Copyright (c) DeepSpeed Team.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
"""DeepSeek Sparse Attention (DSA) with the TileLang training kernels, as a drop-in replacement for
the attention module of GLM-5.2 (transformers ``glm_moe_dsa``).

Why a replacement module and not a registered attention function: the transformers module
decompresses the latent key-value into per-head K and V before it calls the attention interface,
and its indexer builds a dense ``[batch, index_heads, seq, seq]`` float32 score tensor. That path
is quadratic in memory (a 64 GiB indexer allocation at 32K tokens on GLM-5.2) and cannot use the
absorbed form the sparse kernel needs. This module keeps the source module's parameters and
attribute names (state-dict keys and AutoTP patterns are unchanged) and only changes the forward:

* the query is absorbed into the 512-wide latent with this head's slice of ``kv_b_proj``, so
  attention runs on ``[tokens, 1, 512 + 64]`` latent keys, one shared head;
* the indexer runs in blocks of query rows and returns hard top-k token indices;
* the sparse MLA kernel attends over the selected keys only, with a backward pass;
* the output is mapped back to value space with the other slice of ``kv_b_proj``.

The indexer is not trained (a hard top-k has no gradient), which matches transformers, where the
indexer forward is ``@torch.no_grad()``. Training-only, and the module raises rather than computing
something else for: a key-value cache; an attention mask that is not plain causal attention over the
whole sequence (padding, packed sequences) or position ids that are not consecutive; attention dropout;
``output_attentions`` (the sparse kernel never forms the attention weights). Replace every DSA
attention layer of a model or none: a ``"shared"`` layer reuses the previous full layer's indices,
and the two implementations encode them differently for the earliest query rows.

Context parallel (``set_context_parallel_group``): the ranks of a group hold consecutive equal chunks
of one sequence (batch size 1), rank ``r`` the tokens ``[r * S / cp, (r + 1) * S / cp)``, and the
caller passes the matching ``position_ids``. Queries stay local; the latent keys (512 + 64 values per
token) and the indexer keys are gathered over the group, so every query sees its causal keys with
global positions. The gradient of the gathered keys is reduce-scattered back to the owning ranks.
Everything outside attention works per token, so the rest of the model needs no change.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from deepspeed.accelerator import get_accelerator
from deepspeed.runtime.zero.partition_parameters import register_external_parameter
from deepspeed.utils import logger

SUPPORTED_SOURCE_CLASSES = ("GlmMoeDsaAttention", )

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


def _all_gather_sequence(local: torch.Tensor, group, size: int) -> torch.Tensor:
    import deepspeed.comm as dist
    local = local.contiguous()
    full = torch.empty((local.shape[0] * size, *local.shape[1:]), dtype=local.dtype, device=local.device)
    dist.all_gather_into_tensor(full, local, group=group)
    return full


class _GatherSequence(torch.autograd.Function):
    """All-gather along dim 0 over the context-parallel group; the backward reduce-scatters (sums) the gradient."""

    @staticmethod
    def forward(ctx, local: torch.Tensor, group, size: int) -> torch.Tensor:
        ctx.group, ctx.size = group, size
        return _all_gather_sequence(local, group, size)

    @staticmethod
    def backward(ctx, grad_full: torch.Tensor):
        import deepspeed.comm as dist
        grad_full = grad_full.contiguous()
        grad_local = torch.empty((grad_full.shape[0] // ctx.size, *grad_full.shape[1:]),
                                 dtype=grad_full.dtype,
                                 device=grad_full.device)
        dist.reduce_scatter_tensor(grad_local, grad_full, op=dist.ReduceOp.SUM, group=ctx.group)
        return grad_local, None, None


def _gather_sequence(local: torch.Tensor, group, size: int) -> torch.Tensor:
    if size == 1:
        return local
    if not torch.is_grad_enabled() or not local.requires_grad:
        return _all_gather_sequence(local, group, size)
    return _GatherSequence.apply(local, group, size)


def _apply_rotary_interleave(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    """Interleaved rotary embedding on ``[batch, seq, heads, rope_dim]`` tensors.

    The same formula as transformers' ``apply_rotary_pos_emb_interleave`` with ``unsqueeze_dim=2``:
    ``cos`` / ``sin`` are ``cat(freqs, freqs)``, so their first half holds the per-pair angle, and the
    pairs are the even and odd positions of the last dimension.
    """
    cos = cos[..., :cos.shape[-1] // 2].unsqueeze(2)
    sin = sin[..., :sin.shape[-1] // 2].unsqueeze(2)
    q1, q2 = q[..., 0::2], q[..., 1::2]
    k1, k2 = k[..., 0::2], k[..., 1::2]
    q_embed = torch.cat([q1 * cos - q2 * sin, q2 * cos + q1 * sin], dim=-1)
    k_embed = torch.cat([k1 * cos - k2 * sin, k2 * cos + k1 * sin], dim=-1)
    return q_embed, k_embed


def _is_power_of_two(value: int) -> bool:
    return value > 0 and value & (value - 1) == 0


# The (attention_mask, position_ids) objects that last passed _check_causal_only. transformers hands the same two
# tensors to every layer of one forward, so the check (which waits for the GPU) runs once per forward, not per layer.
_LAST_CHECKED = (None, None)


def _check_causal_only(attention_mask: torch.Tensor | None,
                       position_ids: torch.Tensor | None,
                       seq_length: int,
                       block_rows: int = 4096) -> None:
    """The kernels compute causal attention over every whole sequence of the batch. Raise for any input that asks
    for something else: transformers builds a 4-D mask for padding and for packed sequences (detected from position
    ids that restart), and a full causal 4-D mask under eager attention, which is the only mask accepted here."""
    global _LAST_CHECKED
    if _LAST_CHECKED[0] is attention_mask and _LAST_CHECKED[1] is position_ids:
        return
    if position_ids is not None:
        steps = position_ids[..., 1:] - position_ids[..., :-1]
        if not bool((steps == 1).all()):
            raise NotImplementedError("DeepSpeedGlmMoeDsaAttention needs consecutive position ids; packed sequences "
                                      "and gaps are not supported.")
    if attention_mask is not None:
        if attention_mask.dim() != 4 or tuple(attention_mask.shape[-2:]) != (seq_length, seq_length):
            raise NotImplementedError("DeepSpeedGlmMoeDsaAttention supports no attention mask other than plain causal "
                                      f"attention; got a mask of shape {tuple(attention_mask.shape)}.")
        # transformers' float masks are 0 where attention is allowed; bool masks are True there.
        keep = attention_mask if attention_mask.dtype == torch.bool else attention_mask == 0
        positions = torch.arange(seq_length, device=attention_mask.device)
        # Compared in blocks of rows so the reference never covers the whole [seq, seq] square.
        for start in range(0, seq_length, block_rows):
            end = min(start + block_rows, seq_length)
            causal = positions[None, :] <= positions[start:end, None]
            if not bool((keep[..., start:end, :] == causal).all()):
                raise NotImplementedError("DeepSpeedGlmMoeDsaAttention does not support padding or packed-sequence "
                                          "masks: every sequence in the batch must be a full causal sequence.")
    _LAST_CHECKED = (attention_mask, position_ids)


class DeepSpeedGlmMoeDsaAttention(nn.Module):
    """The GLM-5.2 attention layer computed with the DSA kernels. See the module docstring."""

    def __init__(self, source: nn.Module, index_block_rows: int = 8192, warm_up: bool = True):
        super().__init__()
        if getattr(source, "q_a_proj", None) is None:
            raise ValueError("DeepSpeedGlmMoeDsaAttention needs the low-rank query path (q_lora_rank); "
                             "the model has none.")
        config = source.config
        self.config = config
        self.layer_idx = source.layer_idx
        self.num_heads = source.num_heads
        self.qk_nope_head_dim = source.qk_nope_head_dim
        self.qk_rope_head_dim = source.qk_rope_head_dim
        self.kv_lora_rank = source.kv_lora_rank
        self.v_head_dim = source.v_head_dim
        self.qk_head_dim = source.qk_head_dim
        self.scaling = source.scaling
        self.attention_dropout = source.attention_dropout
        self.index_topk = config.index_topk
        self.index_block_rows = index_block_rows
        # Done at the first forward, on the input's device: replace_attention() runs before deepspeed.initialize(),
        # when every local process may still have GPU 0 as its current device.
        self.warm_up = warm_up

        for name, value in (("kv_lora_rank", self.kv_lora_rank), ("qk_rope_head_dim", self.qk_rope_head_dim)):
            # The forward kernel needs a power of two; the backward kernel splits the width into groups of 4.
            if not _is_power_of_two(value) or value < 4:
                raise ValueError(f"the sparse MLA kernels need {name} to be a power of two of at least 4, got {value}")
        if self.index_topk % 64 != 0:
            raise ValueError(f"the sparse MLA kernel needs index_topk to be a multiple of 64, got {self.index_topk}")
        if self.num_heads > 64 and self.num_heads % 64 != 0:
            raise ValueError(f"the sparse MLA kernel needs at most 64 heads or a multiple of 64, got {self.num_heads}")
        if source.indexer is not None and 128 % source.indexer.n_heads != 0:
            raise ValueError(
                f"the indexer kernel needs the index head count to divide 128, got {source.indexer.n_heads}")

        # The same attribute names as the source module, so state-dict keys and AutoTP patterns hold.
        self.q_a_proj = source.q_a_proj
        self.q_a_layernorm = source.q_a_layernorm
        self.q_b_proj = source.q_b_proj
        self.kv_a_proj_with_mqa = source.kv_a_proj_with_mqa
        self.kv_a_layernorm = source.kv_a_layernorm
        self.kv_b_proj = source.kv_b_proj
        self.o_proj = source.o_proj
        self.indexer = source.indexer  # None on a "shared" layer

        # The absorbed form reads kv_b_proj.weight as a tensor instead of calling kv_b_proj, so
        # ZeRO-3 must gather it for this module's own forward and backward.
        register_external_parameter(self, self.kv_b_proj.weight)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None = None,
        past_key_values=None,
        position_ids: torch.Tensor | None = None,
        prev_topk_indices: torch.Tensor | None = None,
        **kwargs,
    ):
        if past_key_values is not None:
            raise NotImplementedError("DeepSpeedGlmMoeDsaAttention is for training: no key-value cache.")
        if self.training and self.attention_dropout:
            raise NotImplementedError("DeepSpeedGlmMoeDsaAttention does not implement attention dropout.")
        if kwargs.get("output_attentions", getattr(self.config, "output_attentions", False)):
            raise NotImplementedError("DeepSpeedGlmMoeDsaAttention cannot return attention weights: the sparse "
                                      "kernel never forms them. Call the model with output_attentions=False.")
        from deepspeed.models.glm_moe_dsa.kernels import sparse_mla

        batch_size, seq_length, _ = hidden_states.shape
        cp_group, cp_size, _ = _context_parallel()
        if cp_size > 1 and batch_size != 1:
            raise NotImplementedError("context parallel DSA attention needs batch size 1 (one sequence per group).")
        _check_causal_only(attention_mask, position_ids, seq_length)
        if self.warm_up:
            _warm_up_once(self, hidden_states.device)
        heads = self.num_heads
        tokens = batch_size * seq_length
        cos, sin = position_embeddings

        q_resid = self.q_a_layernorm(self.q_a_proj(hidden_states))
        q_states = self.q_b_proj(q_resid).view(batch_size, seq_length, heads, self.qk_head_dim)
        q_pass, q_rot = torch.split(q_states, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)
        compressed_kv = self.kv_a_proj_with_mqa(hidden_states)
        kv_pass, k_rot = torch.split(compressed_kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
        kv_pass = self.kv_a_layernorm(kv_pass)
        q_rot, k_rot = _apply_rotary_interleave(q_rot, k_rot.unsqueeze(2), cos, sin)

        # Absorb: q_nope . (W_kc kv) == (W_kc^T q_nope) . kv, per head, so the keys stay in latent form.
        kv_b_weight = self.kv_b_proj.weight.view(heads, self.qk_nope_head_dim + self.v_head_dim, self.kv_lora_rank)
        w_kc = kv_b_weight[:, :self.qk_nope_head_dim, :]
        w_vc = kv_b_weight[:, self.qk_nope_head_dim:, :]
        q_absorbed = torch.einsum("bshd,hdm->bshm", q_pass, w_kc)
        q_full = torch.cat([q_absorbed, q_rot], dim=-1).reshape(tokens, heads, -1).to(torch.bfloat16)
        kv_full = torch.cat([kv_pass, k_rot.squeeze(2)], dim=-1).reshape(tokens, 1, -1).to(torch.bfloat16)
        # Under context parallel the keys of the whole sequence, in rank order = position order.
        kv_full = _gather_sequence(kv_full, cp_group, cp_size)

        if self.indexer is not None:
            topk_indices = self._select_topk(hidden_states, q_resid, cos, sin)
        elif prev_topk_indices is None:
            raise ValueError("Shared DSA layers require top-k indices from a previous full indexer layer.")
        else:
            topk_indices = prev_topk_indices
        # With context parallel the indices are already positions in the gathered sequence (batch size 1).
        global_indices = self._global_indices(topk_indices, seq_length * cp_size)

        out = sparse_mla(q_full, kv_full, global_indices, self.scaling, self.kv_lora_rank)  # [tokens, heads, latent]
        out = torch.einsum("thm,hdm->thd", out.to(w_vc.dtype), w_vc)
        out = out.reshape(batch_size, seq_length, heads * self.v_head_dim)
        return self.o_proj(out), None, topk_indices

    @torch.no_grad()
    def _select_topk(self, hidden_states, q_resid, cos, sin):
        """Top-k key positions per query, ``[batch, seq, topk]`` int32, ``-1`` where a row has fewer keys.

        Same scores as transformers' ``GlmMoeDsaIndexer``: ``sum_h w_h * relu(scale * q_h . k)`` with the
        rotary part interleaved on the first ``qk_rope_head_dim`` dims, causal within each sequence.
        ``scale`` is folded into the weights, which is exact because ``relu(c x) = c relu(x)`` for
        ``c > 0``.
        """
        from deepspeed.models.glm_moe_dsa.kernels import select_topk

        indexer = self.indexer
        batch_size, seq_length, _ = hidden_states.shape
        tokens = batch_size * seq_length
        rope_dim = indexer.qk_rope_head_dim
        q = indexer.wq_b(q_resid).view(batch_size, seq_length, indexer.n_heads, indexer.head_dim)
        q_rot, q_pass = torch.split(q, [rope_dim, indexer.head_dim - rope_dim], dim=-1)
        k = indexer.k_norm(indexer.wk(hidden_states)).unsqueeze(2)
        k_rot, k_pass = torch.split(k, [rope_dim, indexer.head_dim - rope_dim], dim=-1)
        q_rot, k_rot = _apply_rotary_interleave(q_rot, k_rot, cos, sin)
        index_q = torch.cat([q_rot, q_pass], dim=-1).reshape(tokens, indexer.n_heads, indexer.head_dim)
        index_k = torch.cat([k_rot, k_pass], dim=-1).reshape(tokens, indexer.head_dim)
        weights = indexer.weights_proj(hidden_states.to(indexer.weights_proj.weight.dtype)).float()
        weights = weights.reshape(tokens, indexer.n_heads) * (indexer.n_heads**-0.5 * indexer.softmax_scale)

        cp_group, cp_size, cp_rank = _context_parallel()
        if cp_size > 1:
            # Local queries against the keys of the whole sequence: row i of this rank is position
            # cp_rank * seq_length + i, and may see every key up to and including itself.
            index_k = _gather_sequence(index_k.contiguous(), cp_group, cp_size)
            positions = cp_rank * seq_length + torch.arange(tokens, device=hidden_states.device, dtype=torch.int32)
            sequence_start = torch.zeros_like(positions)
        else:
            positions = torch.arange(tokens, device=hidden_states.device, dtype=torch.int32)
            sequence_start = (positions // seq_length) * seq_length
        selected = select_topk(
            index_q.to(torch.bfloat16).contiguous(),
            index_k.to(torch.bfloat16).contiguous(), weights.contiguous(), sequence_start, positions + 1,
            self.index_topk, self.index_block_rows)
        local = torch.where(selected >= 0, selected - sequence_start[:, None], selected)
        return local.view(batch_size, seq_length, self.index_topk)

    @staticmethod
    def _global_indices(topk_indices: torch.Tensor, seq_length: int) -> torch.Tensor:
        """``[batch, seq, topk]`` positions within each sequence to ``[tokens, 1, topk]`` token indices.

        ``seq_length`` is the length of the sequence the keys cover: under context parallel the whole
        gathered sequence (batch size 1, so no offset is added).
        """
        batch_size, rows = topk_indices.shape[0], topk_indices.shape[1]
        offsets = (torch.arange(batch_size, device=topk_indices.device) * seq_length)[:, None, None]
        indices = topk_indices.to(torch.int64)
        indices = torch.where(indices >= 0, indices + offsets, indices)
        return indices.to(torch.int32).reshape(batch_size * rows, 1, -1).contiguous()


_WARMED_UP = set()


def _warm_up_once(attention: DeepSpeedGlmMoeDsaAttention, device: torch.device) -> None:
    """Run ``_warm_up_kernels`` once per device and kernel shape, then never again for this module. Only the
    128-thread backward kernel needs it (see ``_warm_up_kernels``)."""
    from deepspeed.models.glm_moe_dsa.kernels.sparse_mla_bwd import backward_threads

    attention.warm_up = False
    if backward_threads(attention.num_heads) != 128:
        return
    config = attention.config
    key = (str(device), attention.num_heads, attention.kv_lora_rank, attention.qk_rope_head_dim, attention.index_topk,
           config.index_n_heads, config.index_head_dim)
    if key not in _WARMED_UP:
        # No autograd: under activation checkpointing this runs inside the first forward but not in its
        # recomputation, and any tensor autograd saved here would make the two passes save different numbers
        # of tensors.
        with torch.no_grad():
            _warm_up_kernels(attention, device)
        _WARMED_UP.add(key)


def _warm_up_kernels(attention: DeepSpeedGlmMoeDsaAttention, device: torch.device) -> None:
    """Compile, load and launch every DSA kernel once, forward and backward, for this module's shapes.

    With 128 threads (head blocks under 64 heads) the sparse MLA backward kernel spills registers to a
    2,896-byte stack per thread, so its first launch makes the CUDA driver reserve local memory for every
    thread the device can hold (about 0.73 GiB on an H200) outside PyTorch's allocator. The 256-thread
    kernel of 64-head blocks does not spill (its first launch at 32,768 tokens takes 0.002 GiB), so it
    gets no warm-up. When that first launch comes mid-training, the
    allocator's cache may already hold the rest of the device and the launch fails with
    CUDA_ERROR_OUT_OF_MEMORY (GLM-5.2, 78 layers, 65,536 tokens per GPU, 64 H200s). The driver keeps a
    local-memory reservation once it has made it, and loaded kernels stay loaded, so launching each
    kernel once here, while the device is still empty, settles both. Head counts, head dims and top-k
    are compiled into every kernel. Sequence lengths are dynamic in the forward and indexer kernels,
    so the short sequence here builds the forward kernels training uses. The backward kernel takes the
    batch size and both sequence lengths as compile-time values, so training's first backward still
    builds its own backward kernel; the launch here only makes the local-memory reservation.
    """
    from deepspeed.models.glm_moe_dsa.kernels import select_topk
    from deepspeed.models.glm_moe_dsa.kernels.sparse_mla_bwd import sparse_mla_bwd
    from deepspeed.models.glm_moe_dsa.kernels.sparse_mla_fwd import sparse_mla_fwd_interface

    config = attention.config
    tokens = 128
    rope = attention.qk_rope_head_dim
    latent = attention.kv_lora_rank
    index_heads = config.index_n_heads
    index_dim = config.index_head_dim

    positions = torch.arange(tokens, device=device, dtype=torch.int32)
    index_q = torch.randn(tokens, index_heads, index_dim, device=device, dtype=torch.bfloat16)
    index_k = torch.randn(tokens, index_dim, device=device, dtype=torch.bfloat16)
    weights = torch.rand(tokens, index_heads, device=device)
    indices = select_topk(index_q, index_k, weights, torch.zeros_like(positions), positions + 1,
                          attention.index_topk).view(tokens, 1, -1)

    q = torch.randn(tokens, attention.num_heads, latent + rope, device=device, dtype=torch.bfloat16)
    kv = torch.randn(tokens, 1, latent + rope, device=device, dtype=torch.bfloat16)
    # The kernels directly, not through autograd (see _warm_up_once).
    out, lse = sparse_mla_fwd_interface(q, kv, indices, attention.scaling, latent)
    sparse_mla_bwd(q, kv, out, torch.randn_like(out), indices, lse, attention.scaling, latent)
    get_accelerator().synchronize(device.index)


def replace_attention(model: nn.Module, index_block_rows: int = 8192, warm_up: bool = True) -> int:
    """Replace every DSA attention module of ``model`` with :class:`DeepSpeedGlmMoeDsaAttention`.

    Returns the number of replaced modules. Call it after the model is built and before
    ``deepspeed.initialize``; the parameters are reused, not copied, so it also works on a model
    built under ``deepspeed.zero.Init``. With ``warm_up`` (the default) every kernel is also
    launched once at the first forward, on that forward's device, when the backward kernel uses 128
    threads (fewer than 64 heads per block); see ``_warm_up_kernels`` for why that must happen before
    training fills the device.
    """
    try:
        import tilelang  # noqa: F401
    except ImportError as err:
        raise ImportError("The GLM-5.2 DSA kernels need TileLang: pip install tilelang") from err

    replaced = []
    for name, module in list(model.named_modules()):
        if type(module).__name__ not in SUPPORTED_SOURCE_CLASSES:
            continue
        parent_name, _, attr = name.rpartition(".")
        parent = model.get_submodule(parent_name) if parent_name else model
        attention = DeepSpeedGlmMoeDsaAttention(module, index_block_rows=index_block_rows, warm_up=warm_up)
        setattr(parent, attr, attention)
        replaced.append(attention)
    logger.info(f"DSA kernels: replaced {len(replaced)} attention module(s) with DeepSpeedGlmMoeDsaAttention")
    return len(replaced)
