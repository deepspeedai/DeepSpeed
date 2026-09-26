# Copyright (c) DeepSpeed Team.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
"""DeepSeek Sparse Attention (DSA) training kernels in TileLang.

Two pieces, both in the absorbed multi-head latent attention (MLA) form:

* :func:`select_topk`: the lightning indexer. Dense fp32 scores of every query against every key,
  in blocks of query rows, then the top-k keys per query. Returns token indices, ``-1`` where a
  query has fewer than ``topk`` visible keys.
* :func:`sparse_mla`: attention of each query over only its selected keys, on the latent
  ``[tokens, 1, dim + rope_dim]`` key-value tensor, with a backward pass. Memory and compute are
  linear in the sequence length times ``topk``.

The kernels are adapted from the TileLang ``deepseek_v32`` examples, as used by the GLM-5 plugin
of THUDM/slime; the Megatron sequence-parallel and context-parallel handling of that plugin is not
part of them. They require bf16 inputs on a CUDA device. Importing this package imports TileLang.
"""

from deepspeed.models.glm_moe_dsa.kernels.indexer import select_topk
from deepspeed.models.glm_moe_dsa.kernels.sparse_mla import sparse_mla

__all__ = ["select_topk", "sparse_mla"]
