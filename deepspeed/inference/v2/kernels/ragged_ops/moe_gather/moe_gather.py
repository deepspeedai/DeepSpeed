# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import torch

from ... import DSKernelBase
from ....inference_utils import DtypeEnum
from deepspeed.ops.op_builder import RaggedOpsBuilder


class MoEGather(DSKernelBase):
    """
    CUDA implementation of MoE gather. This will bring the tokens back
    to their original indices and perform the output scaling.
    """

    supported_dtypes = [DtypeEnum.fp16, DtypeEnum.bf16]

    def __init__(self, dtype: DtypeEnum, channels: int, normalize_scores: bool = False) -> None:

        if not isinstance(dtype, DtypeEnum):
            dtype = DtypeEnum(dtype)

        if dtype not in MoEGather.supported_dtypes:
            raise RuntimeError(f"Unsupported dtype {dtype}")

        if channels % 8 != 0:
            raise RuntimeError(f"Channels {channels} must be divisible by 8")

        inf_module = RaggedOpsBuilder().load()
        self.kernel = inf_module.moe_gather
        self.normalize_scores = normalize_scores

    def __call__(self, layer_output: torch.Tensor, moe_output: torch.Tensor, scores: torch.Tensor,
                 mapped_slots: torch.Tensor, expert_counts: torch.Tensor) -> torch.Tensor:
        """
        Reorders the moe_output tokens into their original order and scales them by their
        gating scale. This will be a no-op for padded tokens.

        Arguments:
            layer_output (torch.Tensor): The output of the layer of shape [n_tokens, hidden_size]. This has been scaled appropriately.
            moe_output (torch.Tensor): The output of the MoE of shape [n_tokens * n_top_k, hidden_size].
            scores (torch.Tensor): The gating scores of shape [n_tokens].
            mapped_slots (torch.Tensor): The index of the token in the expert's input of shape [n_tokens, n_top_k]. The indices of token ``i`` in layer_output is ``mapped_slots[i]``.
            expert_counts (torch.Tensor): The number of tokens assigned to each expert of shape [n_experts]. This is passed to fuse the clearing of this data structure into the gather.

        Returns:
            layer_output
        """
        self.kernel(layer_output, moe_output, scores, mapped_slots, expert_counts, self.normalize_scores)
        return layer_output
