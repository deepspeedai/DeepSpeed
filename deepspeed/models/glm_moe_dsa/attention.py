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
indexer forward is ``@torch.no_grad()``. Training-only: no key-value cache, no padding mask (every
sequence in the batch is a full causal sequence), attention dropout must be 0. Replace every DSA
attention layer of a model or none: a ``"shared"`` layer reuses the previous full layer's indices,
and the two implementations encode them differently for the earliest query rows.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from deepspeed.accelerator import get_accelerator
from deepspeed.runtime.zero.partition_parameters import register_external_parameter
from deepspeed.utils import logger

SUPPORTED_SOURCE_CLASSES = ("GlmMoeDsaAttention", )


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


class DeepSpeedGlmMoeDsaAttention(nn.Module):
    """The GLM-5.2 attention layer computed with the DSA kernels. See the module docstring."""

    def __init__(self, source: nn.Module, index_block_rows: int = 8192):
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

        for name, value in (("kv_lora_rank", self.kv_lora_rank), ("qk_rope_head_dim", self.qk_rope_head_dim)):
            if not _is_power_of_two(value):
                raise ValueError(f"the sparse MLA kernel needs {name} to be a power of two, got {value}")
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
        from deepspeed.models.glm_moe_dsa.kernels import sparse_mla

        batch_size, seq_length, _ = hidden_states.shape
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

        if self.indexer is not None:
            topk_indices = self._select_topk(hidden_states, q_resid, cos, sin)
        elif prev_topk_indices is None:
            raise ValueError("Shared DSA layers require top-k indices from a previous full indexer layer.")
        else:
            topk_indices = prev_topk_indices
        global_indices = self._global_indices(topk_indices, seq_length)

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
        """``[batch, seq, topk]`` positions within each sequence to ``[tokens, 1, topk]`` token indices."""
        batch_size = topk_indices.shape[0]
        offsets = (torch.arange(batch_size, device=topk_indices.device) * seq_length)[:, None, None]
        indices = topk_indices.to(torch.int64)
        indices = torch.where(indices >= 0, indices + offsets, indices)
        return indices.to(torch.int32).reshape(batch_size * seq_length, 1, -1).contiguous()


def _warm_up_kernels(attention: DeepSpeedGlmMoeDsaAttention) -> None:
    """Compile, load and launch every DSA kernel once, forward and backward, for this module's shapes.

    The sparse MLA backward kernel spills registers to a 2,896-byte stack per thread, so its first
    launch makes the CUDA driver reserve local memory for every thread the device can hold (about
    0.73 GiB on an H200) outside PyTorch's allocator. When that first launch comes mid-training, the
    allocator's cache may already hold the rest of the device and the launch fails with
    CUDA_ERROR_OUT_OF_MEMORY (GLM-5.2, 78 layers, 65,536 tokens per GPU, 64 H200s). The driver keeps a
    local-memory reservation once it has made it, and loaded kernels stay loaded, so launching each
    kernel once here, while the device is still empty, settles both. Sequence lengths are dynamic in
    the kernels; head counts, head dims and top-k are compiled in, so a short sequence builds exactly
    the kernels training uses.
    """
    from deepspeed.models.glm_moe_dsa.kernels import select_topk, sparse_mla

    config = attention.config
    device = get_accelerator().current_device_name()
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
    q.requires_grad_(True)
    kv.requires_grad_(True)
    out = sparse_mla(q, kv, indices, attention.scaling, latent)
    out.backward(torch.randn_like(out))
    get_accelerator().synchronize()


def replace_attention(model: nn.Module, index_block_rows: int = 8192, warm_up: bool = True) -> int:
    """Replace every DSA attention module of ``model`` with :class:`DeepSpeedGlmMoeDsaAttention`.

    Returns the number of replaced modules. Call it after the model is built and before
    ``deepspeed.initialize``; the parameters are reused, not copied, so it also works on a model
    built under ``deepspeed.zero.Init``. With ``warm_up`` (the default) every kernel is also
    compiled and launched once on the current accelerator; see ``_warm_up_kernels`` for why that
    must happen before training fills the device.
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
        attention = DeepSpeedGlmMoeDsaAttention(module, index_block_rows=index_block_rows)
        setattr(parent, attr, attention)
        replaced.append(attention)
    logger.info(f"DSA kernels: replaced {len(replaced)} attention module(s) with DeepSpeedGlmMoeDsaAttention")

    if warm_up and replaced:
        warmed = set()
        for attention in replaced:
            shape = (attention.num_heads, attention.kv_lora_rank, attention.qk_rope_head_dim, attention.index_topk,
                     attention.config.index_n_heads, attention.config.index_head_dim)
            if shape not in warmed:
                _warm_up_kernels(attention)
                warmed.add(shape)
    return len(replaced)
