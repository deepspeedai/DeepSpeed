# Copyright (c) DeepSpeed Team.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
"""MiniMax-M3 (transformers ``minimax_m3_vl``): block-sparse attention whose memory and compute grow
linearly with the sequence length, in Triton (``selected_attention``) or plain PyTorch
(``block_sparse_attention``), optionally with each sequence split over a context-parallel group.

Call :func:`replace_attention` on the built model before ``deepspeed.initialize``.
"""

from deepspeed.models.minimax_m3_vl.attention import (DeepSpeedMiniMaxM3VLAttention,
                                                      DeepSpeedMiniMaxM3VLDenseAttention, replace_attention,
                                                      set_context_parallel_group)
from deepspeed.models.minimax_m3_vl.block_sparse_attention import block_sparse_attention
from deepspeed.models.minimax_m3_vl.index_kernel import index_block_max
from deepspeed.models.minimax_m3_vl.selected_attention import selected_attention

__all__ = [
    "DeepSpeedMiniMaxM3VLAttention", "DeepSpeedMiniMaxM3VLDenseAttention", "block_sparse_attention", "index_block_max",
    "replace_attention", "selected_attention", "set_context_parallel_group"
]
