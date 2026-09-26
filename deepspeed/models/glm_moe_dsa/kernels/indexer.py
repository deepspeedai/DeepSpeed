# Copyright (c) DeepSpeed Team.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
"""Top-k key selection with the lightning indexer kernel."""

import torch

from deepspeed.models.glm_moe_dsa.kernels.indexer_fwd import indexer_logits


@torch.no_grad()
def select_topk(index_q: torch.Tensor,
                index_k: torch.Tensor,
                weights: torch.Tensor,
                cu_seqlen_ks: torch.Tensor,
                cu_seqlen_ke: torch.Tensor,
                topk: int,
                block_rows: int = 8192) -> torch.Tensor:
    """Select the ``topk`` keys of every query row.

    Runs in blocks of ``block_rows`` query rows so that the float32 logits, ``[block_rows, tokens_kv]``,
    never cover the whole sequence at once (2 GiB per block at 65,536 keys). The indexer itself does
    not train here: the selection is a hard top-k, so no gradient reaches ``index_q``, ``index_k`` or
    ``weights``, which matches transformers' ``@torch.no_grad()`` indexer.

    Args:
        index_q: ``[tokens, heads, index_dim]`` bf16.
        index_k: ``[tokens_kv, index_dim]`` bf16.
        weights: ``[tokens, heads]`` float32 per-head weights with every scale folded in.
        cu_seqlen_ks, cu_seqlen_ke: int32 ``[tokens]``; row ``t`` may select keys in ``[ks[t], ke[t])``.
        topk: number of keys per row.

    Returns:
        ``[tokens, topk]`` int32 token indices into ``index_k``; ``-1`` where a row has fewer than
        ``topk`` visible keys.
    """
    tokens, tokens_kv = index_q.shape[0], index_k.shape[0]
    selected = torch.empty(tokens, topk, dtype=torch.int32, device=index_q.device)
    k = min(topk, tokens_kv)
    for start in range(0, tokens, block_rows):
        end = min(start + block_rows, tokens)
        logits = indexer_logits(index_q[start:end], index_k, weights[start:end], cu_seqlen_ks[start:end],
                                cu_seqlen_ke[start:end])
        scores, indices = logits.topk(k, dim=-1)
        indices = indices.to(torch.int32).masked_fill_(scores == float("-inf"), -1)
        selected[start:end, :k] = indices
        if k < topk:
            selected[start:end, k:] = -1
    return selected
