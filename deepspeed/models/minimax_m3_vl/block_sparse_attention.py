# Copyright (c) DeepSpeed Team.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
"""Block-sparse attention in plain PyTorch, with a forward and a backward pass whose memory and
compute grow linearly with the sequence length.

Each query attends only to the keys of the key blocks selected for it, and only to keys at or
before its own position. The selection is per key-value head: ``block_indices[b, h, t]`` lists the
blocks that the query heads sharing key-value head ``h`` may read at query position ``t``.

The work goes one chunk of query rows at a time. For a chunk, the selected key and value blocks
are gathered into ``[rows, topk * block_size, head_dim]`` tensors and a single softmax runs over
them, so nothing of size ``seq x seq`` is ever built. The forward pass keeps only the output and the
float32 log-sum-exp of every query and head; the backward pass recomputes the probabilities from
them, chunk by chunk, as flash attention does. The key and value gradients are added into float32
buffers. One key receives a gradient from every query that selected its block, which can be every
query in the sequence. Adding those terms in bf16 rounds after every addition; on MiniMax-M3's shapes
at 65,536 tokens that gave the key and value gradients a relative error of 2.2e-2, against 2.4e-3
with float32 buffers.
"""

import torch
import torch.nn.functional as F

from deepspeed.utils.torch import required_torch_version

# torch.bmm(..., out_dtype=torch.float32) takes bf16 inputs directly; before 2.8 they are upcast.
_BMM_HAS_OUT_DTYPE = required_torch_version(min_version=2.8)


def _bmm_float32(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """``a @ b`` with a float32 result, without first copying half-precision inputs to float32."""
    if a.dtype == torch.float32:
        return torch.bmm(a, b)
    if _BMM_HAS_OUT_DTYPE and a.device.type == "cuda":  # the float32-output overload is CUDA only
        return torch.bmm(a, b, out_dtype=torch.float32)
    return torch.bmm(a.float(), b.float())


def _block_rows(x: torch.Tensor, num_blocks: int, block_size: int) -> torch.Tensor:
    """``[batch, kv_heads, seq, dim]`` to ``[batch * kv_heads * num_blocks, block_size * dim]``,
    padding the sequence up to whole blocks. Row ``(b * kv_heads + h) * num_blocks + n`` holds block
    ``n`` of head ``h`` of sequence ``b``."""
    batch, kv_heads, seq_len, dim = x.shape
    pad = num_blocks * block_size - seq_len
    x = F.pad(x, (0, 0, 0, pad)) if pad else x.contiguous()
    return x.view(batch * kv_heads * num_blocks, block_size * dim)


def _chunk_selection(block_indices: torch.Tensor, positions: torch.Tensor, num_blocks: int, block_size: int,
                     kv_len: int):
    """The rows of :func:`_block_rows` to gather for one chunk of queries, and which gathered keys
    each query may see.

    Args:
        block_indices: ``[batch, kv_heads, rows, topk]`` selected block ids, ``-1`` for an unused slot.
        positions: ``[batch, rows]`` position of every query.

    Returns:
        ``gather_rows`` ``[batch * kv_heads * rows * topk]`` and ``visible``
        ``[batch * kv_heads * rows, 1, topk * block_size]`` (True where the query may attend).
    """
    batch, kv_heads, rows, topk = block_indices.shape
    device = block_indices.device
    # An unused slot gathers block 0 and is then hidden by ``visible``.
    blocks = block_indices.clamp(min=0)
    head_offset = torch.arange(batch * kv_heads, device=device).view(batch, kv_heads, 1, 1) * num_blocks
    gather_rows = (head_offset + blocks).reshape(-1)

    key_positions = blocks.unsqueeze(-1) * block_size + torch.arange(block_size, device=device)
    causal = key_positions <= positions[:, None, :, None, None]
    visible = (block_indices >= 0).unsqueeze(-1) & causal & (key_positions < kv_len)
    return gather_rows, visible.reshape(batch * kv_heads * rows, 1, topk * block_size)


def _to_groups(x: torch.Tensor, kv_heads: int) -> torch.Tensor:
    """``[batch, rows, heads, dim]`` to ``[batch * kv_heads * rows, group, dim]``. Query heads
    ``h * group`` to ``(h + 1) * group - 1`` share key-value head ``h``, the grouping of transformers'
    ``repeat_kv``."""
    batch, rows, heads, dim = x.shape
    group = heads // kv_heads
    x = x.reshape(batch, rows, kv_heads, group, dim).permute(0, 2, 1, 3, 4)
    return x.reshape(batch * kv_heads * rows, group, dim)


def _from_groups(x: torch.Tensor, batch: int, kv_heads: int) -> torch.Tensor:
    """The inverse of :func:`_to_groups`: ``[batch * kv_heads * rows, group, dim]`` to
    ``[batch, rows, heads, dim]``."""
    _, group, dim = x.shape
    rows = x.shape[0] // (batch * kv_heads)
    x = x.view(batch, kv_heads, rows, group, dim).permute(0, 2, 1, 3, 4)
    return x.reshape(batch, rows, kv_heads * group, dim)


class _BlockSparseAttention(torch.autograd.Function):

    @staticmethod
    def forward(ctx, q, k, v, block_indices, positions, block_size, scale, query_chunk):
        batch, heads, seq_len, dim = q.shape
        kv_heads, kv_len = k.shape[1], k.shape[2]
        group = heads // kv_heads
        num_blocks = -(-kv_len // block_size)
        k_rows = _block_rows(k, num_blocks, block_size)
        v_rows = _block_rows(v, num_blocks, block_size)

        out = q.new_empty(batch, seq_len, heads, dim)
        lse = torch.empty(batch, kv_heads, seq_len, group, dtype=torch.float32, device=q.device)
        for start in range(0, seq_len, query_chunk):
            end = min(start + query_chunk, seq_len)
            gather_rows, visible = _chunk_selection(block_indices[:, :, start:end], positions[:, start:end],
                                                    num_blocks, block_size, kv_len)
            k_sel = k_rows.index_select(0, gather_rows).view(visible.shape[0], -1, dim)
            v_sel = v_rows.index_select(0, gather_rows).view(visible.shape[0], -1, dim)
            q_c = _to_groups(q[:, :, start:end].transpose(1, 2), kv_heads)

            scores = _bmm_float32(q_c, k_sel.transpose(1, 2)).mul_(scale)
            scores.masked_fill_(~visible, float("-inf"))
            lse_c = torch.logsumexp(scores, dim=-1)
            probs = scores.sub_(lse_c.unsqueeze(-1)).exp_()
            out[:, start:end] = _from_groups(torch.bmm(probs.to(v.dtype), v_sel), batch, kv_heads)
            lse[:, :, start:end] = lse_c.view(batch, kv_heads, end - start, group)

        ctx.save_for_backward(q, k, v, block_indices, positions, out, lse)
        ctx.block_size = block_size
        ctx.scale = scale
        ctx.query_chunk = query_chunk
        return out

    @staticmethod
    def backward(ctx, grad_out):
        q, k, v, block_indices, positions, out, lse = ctx.saved_tensors
        block_size, scale, query_chunk = ctx.block_size, ctx.scale, ctx.query_chunk
        batch, heads, seq_len, dim = q.shape
        kv_heads, kv_len = k.shape[1], k.shape[2]
        group = heads // kv_heads
        num_blocks = -(-kv_len // block_size)
        k_rows = _block_rows(k, num_blocks, block_size)
        v_rows = _block_rows(v, num_blocks, block_size)

        grad_q = torch.empty(batch, seq_len, heads, dim, dtype=q.dtype, device=q.device)
        grad_k_rows = torch.zeros(k_rows.shape, dtype=torch.float32, device=k.device)
        grad_v_rows = torch.zeros(v_rows.shape, dtype=torch.float32, device=v.device)
        for start in range(0, seq_len, query_chunk):
            end = min(start + query_chunk, seq_len)
            gather_rows, visible = _chunk_selection(block_indices[:, :, start:end], positions[:, start:end],
                                                    num_blocks, block_size, kv_len)
            k_sel = k_rows.index_select(0, gather_rows).view(visible.shape[0], -1, dim)
            v_sel = v_rows.index_select(0, gather_rows).view(visible.shape[0], -1, dim)
            q_c = _to_groups(q[:, :, start:end].transpose(1, 2), kv_heads)
            grad_out_c = _to_groups(grad_out[:, start:end], kv_heads)
            out_c = _to_groups(out[:, start:end], kv_heads)

            scores = _bmm_float32(q_c, k_sel.transpose(1, 2)).mul_(scale)
            scores.masked_fill_(~visible, float("-inf"))
            lse_c = lse[:, :, start:end].reshape(-1, group, 1)
            probs = scores.sub_(lse_c).exp_()

            grad_v_sel = _bmm_float32(probs.to(v.dtype).transpose(1, 2), grad_out_c)
            grad_v_rows.index_add_(0, gather_rows, grad_v_sel.view(-1, block_size * dim))
            del grad_v_sel

            # Softmax backward: dS = P * (dP - rowsum(dO * O)), times the scale applied to the scores.
            row_dot = (grad_out_c.float() * out_c.float()).sum(dim=-1, keepdim=True)
            grad_probs = _bmm_float32(grad_out_c, v_sel.transpose(1, 2))
            grad_scores = probs.mul_(grad_probs.sub_(row_dot)).mul_(scale).to(q.dtype)
            del grad_probs

            grad_q[:, start:end] = _from_groups(torch.bmm(grad_scores, k_sel), batch, kv_heads)
            grad_k_sel = _bmm_float32(grad_scores.transpose(1, 2), q_c)
            grad_k_rows.index_add_(0, gather_rows, grad_k_sel.view(-1, block_size * dim))
            del grad_k_sel

        padded_len = num_blocks * block_size
        grad_k = grad_k_rows.view(batch, kv_heads, padded_len, dim)[:, :, :kv_len].to(k.dtype)
        grad_v = grad_v_rows.view(batch, kv_heads, padded_len, dim)[:, :, :kv_len].to(v.dtype)
        return grad_q.transpose(1, 2), grad_k, grad_v, None, None, None, None, None


def block_sparse_attention(q: torch.Tensor,
                           k: torch.Tensor,
                           v: torch.Tensor,
                           block_indices: torch.Tensor,
                           positions: torch.Tensor,
                           block_size: int,
                           scale: float,
                           query_chunk: int = 512) -> torch.Tensor:
    """Attention of every query over the keys of its selected key blocks, causal by position.

    Args:
        q: ``[batch, heads, seq, head_dim]``.
        k, v: ``[batch, kv_heads, kv_seq, head_dim]``; ``heads`` must be a multiple of ``kv_heads``.
        block_indices: ``[batch, kv_heads, seq, topk]`` integer ids of the selected key blocks of
            ``block_size`` keys, ``-1`` for an unused slot. The ids of one query must be distinct,
            and every query must be able to see at least one key.
        positions: ``[batch, seq]`` position of every query; a query sees the keys at positions
            ``<=`` its own.
        block_size: keys per block.
        scale: softmax scale applied to ``q . k``.
        query_chunk: query rows per step. Peak memory of one step is about
            ``batch * kv_heads * query_chunk * topk * block_size * head_dim`` elements, times 2 bytes
            for each gathered key and value tensor and 4 bytes for each float32 gradient.

    Returns:
        ``[batch, seq, heads, head_dim]``, the layout the output projection reads.
    """
    return _BlockSparseAttention.apply(q, k, v, block_indices, positions, block_size, scale, query_chunk)
