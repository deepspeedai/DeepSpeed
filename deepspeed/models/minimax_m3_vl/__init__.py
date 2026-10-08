# Copyright (c) DeepSpeed Team.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
"""MiniMax-M3 (transformers ``minimax_m3_vl``): block-sparse attention whose memory and compute grow
linearly with the sequence length, in plain PyTorch.

Call :func:`replace_attention` on the built model before ``deepspeed.initialize``.
"""

from deepspeed.models.minimax_m3_vl.attention import DeepSpeedMiniMaxM3VLAttention, replace_attention
from deepspeed.models.minimax_m3_vl.block_sparse_attention import block_sparse_attention

__all__ = ["DeepSpeedMiniMaxM3VLAttention", "block_sparse_attention", "replace_attention"]
