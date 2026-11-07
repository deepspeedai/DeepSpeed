# Copyright (c) DeepSpeed Team.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
"""The MiniMax-M3 indexer's per-block maximum scores in one Triton kernel.

The indexer scores every query against every key (``q . k``, one index head at a time), masks the keys after the
query, and keeps the maximum over each block of ``block_size`` keys; the top blocks are then selected. Computed with
PyTorch, the scores of one 2,048-query chunk against 65,536 keys are a 2 GiB float32 tensor that is written, copied
by the mask and read again by the maximum. Here each program computes the scores of a tile of queries against one
key block at a time in registers and writes only the block maxima. Blocks that are in the future of every query of
the tile are written as ``-inf`` without being computed.

The inputs are half-precision values, so ``tl.dot`` with float32 accumulation forms every product exactly, as the
float32 GEMM does; only the order of the float32 additions differs (as with ``torch.bmm(..., out_dtype=float32)``),
so near-ties between blocks may be ordered differently.
"""

import torch

from deepspeed.ops.triton_ops._triton import _TRITON_AVAILABLE, tl, triton

if _TRITON_AVAILABLE:

    @triton.jit
    def _index_block_max_kernel(Q, K, QPOS, OUT, stride_qb, stride_qh, stride_qs, stride_kb, stride_ks, stride_pb,
                                stride_ob, stride_oh, stride_os, S_Q, S_K, NB, H: tl.constexpr, D: tl.constexpr,
                                BM: tl.constexpr, BLOCK: tl.constexpr, NB_PER_PROG: tl.constexpr):
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)
        pid_bh = tl.program_id(2)
        b = (pid_bh // H).to(tl.int64)
        h = pid_bh % H
        rows = pid_m * BM + tl.arange(0, BM)
        row_ok = rows < S_Q
        dims = tl.arange(0, D)
        q = tl.load(Q + b * stride_qb + h * stride_qh + rows[:, None].to(tl.int64) * stride_qs + dims[None, :],
                    mask=row_ok[:, None],
                    other=0.0)
        qpos = tl.load(QPOS + b * stride_pb + rows, mask=row_ok, other=-1)
        last_pos = tl.max(qpos, axis=0)
        out_ptrs = OUT + b * stride_ob + h * stride_oh + rows.to(tl.int64) * stride_os
        for j in range(NB_PER_PROG):
            blk = pid_n * NB_PER_PROG + j
            if blk < NB:
                first_key = blk * BLOCK
                if first_key > last_pos:
                    tl.store(out_ptrs + blk, tl.full((BM, ), float("-inf"), tl.float32), mask=row_ok)
                else:
                    keys = first_key + tl.arange(0, BLOCK)
                    k = tl.load(K + b * stride_kb + keys[None, :].to(tl.int64) * stride_ks + dims[:, None],
                                mask=keys[None, :] < S_K,
                                other=0.0)
                    scores = tl.dot(q, k)
                    visible = (keys[None, :] <= qpos[:, None]) & (keys[None, :] < S_K)
                    scores = tl.where(visible, scores, float("-inf"))
                    tl.store(out_ptrs + blk, tl.max(scores, axis=1), mask=row_ok)


def index_block_max(q: torch.Tensor, keys: torch.Tensor, positions: torch.Tensor, block_size: int) -> torch.Tensor:
    """Per-block maximum of the causal indexer scores.

    ``q`` ``[batch, heads, queries, dim]`` and ``keys`` ``[batch, 1, key_len, dim]`` in bf16 or fp16 on the GPU;
    ``positions`` ``[batch, queries]``: each query's position in the key sequence (key ``j`` is visible to the query
    if ``j <= position``). Returns ``[batch, heads, queries, ceil(key_len / block_size)]`` float32, ``-inf`` for a
    block with no visible key.
    """
    batch, heads, num_queries, dim = q.shape
    key_len = keys.shape[2]
    if keys.shape[0] != batch or keys.shape[1] != 1 or keys.shape[3] != dim or keys.dtype != q.dtype:
        raise ValueError(f"keys {tuple(keys.shape)} {keys.dtype} do not match q {tuple(q.shape)} {q.dtype}")
    if dim & (dim - 1) or block_size & (block_size - 1):
        raise ValueError(f"head dim {dim} and block size {block_size} must be powers of 2")
    q = q.contiguous()
    keys = keys.contiguous()
    positions = positions.to(torch.int64).contiguous()
    num_blocks = -(-key_len // block_size)
    out = torch.empty(batch, heads, num_queries, num_blocks, device=q.device, dtype=torch.float32)
    tile_rows, blocks_per_program = 128, 8
    grid = (triton.cdiv(num_queries, tile_rows), triton.cdiv(num_blocks, blocks_per_program), batch * heads)
    _index_block_max_kernel[grid](q,
                                  keys,
                                  positions,
                                  out,
                                  q.stride(0),
                                  q.stride(1),
                                  q.stride(2),
                                  keys.stride(0),
                                  keys.stride(2),
                                  positions.stride(0),
                                  out.stride(0),
                                  out.stride(1),
                                  out.stride(2),
                                  num_queries,
                                  key_len,
                                  num_blocks,
                                  H=heads,
                                  D=dim,
                                  BM=tile_rows,
                                  BLOCK=block_size,
                                  NB_PER_PROG=blocks_per_program)
    return out
