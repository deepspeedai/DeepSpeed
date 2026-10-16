# Copyright (c) DeepSpeed Team.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
"""Autograd wrapper around the sparse MLA kernels."""

import torch

from deepspeed.models.glm_moe_dsa.kernels.sparse_mla_bwd import sparse_mla_bwd
from deepspeed.models.glm_moe_dsa.kernels.sparse_mla_fwd import sparse_mla_fwd_interface


class _SparseMLAFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, q, kv, indices, scale, d_v):
        q, kv, indices = q.contiguous(), kv.contiguous(), indices.contiguous()
        out, lse = sparse_mla_fwd_interface(q, kv, indices, sm_scale=scale, d_v=d_v)
        ctx.save_for_backward(q, kv, indices, out, lse)
        ctx.scale = scale
        ctx.d_v = d_v
        return out

    @staticmethod
    def backward(ctx, grad_out):
        q, kv, indices, out, lse = ctx.saved_tensors
        dq, dkv = sparse_mla_bwd(q, kv, out, grad_out.contiguous(), indices, lse, sm_scale=ctx.scale, d_v=ctx.d_v)
        return dq, dkv, None, None, None


def sparse_mla(q: torch.Tensor, kv: torch.Tensor, indices: torch.Tensor, scale: float, d_v: int = 512) -> torch.Tensor:
    """Attention of every query over its selected latent keys, in the absorbed MLA form.

    Args:
        q: ``[tokens, heads, d_v + rope_dim]`` bf16, the query absorbed into the latent space with the
            rotary part appended.
        kv: ``[tokens_kv, 1, d_v + rope_dim]`` bf16, the latent key-value with the rotary key appended.
        indices: ``[tokens, 1, topk]`` int32 token indices into ``kv``; ``-1`` marks an unused slot.
            ``topk`` must be a multiple of 64.
        scale: the softmax scale (the model's ``1 / sqrt(qk_head_dim)`` with any YaRN factor).
        d_v: the latent width, a power of two; the rotary width must be one as well.

    Returns:
        ``[tokens, heads, d_v]`` bf16. The caller multiplies by the per-head value projection to get
        the value-space output.
    """
    return _SparseMLAFunction.apply(q, kv, indices, scale, d_v)
