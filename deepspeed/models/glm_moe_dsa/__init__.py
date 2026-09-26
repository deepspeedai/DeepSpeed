# Copyright (c) DeepSpeed Team.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
"""GLM-5.2 (transformers ``glm_moe_dsa``): DeepSeek Sparse Attention with TileLang kernels.

Call :func:`replace_attention` on the built model before ``deepspeed.initialize``. It needs
TileLang (``pip install tilelang``) and a CUDA device; importing this package needs neither.
"""

from deepspeed.models.glm_moe_dsa.attention import DeepSpeedGlmMoeDsaAttention, replace_attention

__all__ = ["DeepSpeedGlmMoeDsaAttention", "replace_attention"]
