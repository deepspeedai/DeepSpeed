# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import torch

from typing import Tuple

from ... import DSKernelBase
from ....inference_utils import DtypeEnum
from deepspeed.ops.op_builder import RaggedOpsBuilder


class MoEScatter(DSKernelBase):
    """
    CUDA implementation of MoE scatter
    """

    supported_dtypes = [DtypeEnum.fp16, DtypeEnum.bf16]

    def __init__(self, dtype: DtypeEnum, channels: int) -> None:

        if not isinstance(dtype, DtypeEnum):
            dtype = DtypeEnum(dtype)

        if dtype not in MoEScatter.supported_dtypes:
            raise RuntimeError(f"Unsupported dtype {dtype}")

        if channels % 8 != 0:
            raise RuntimeError(f"Channels {channels} must be divisible by 8")

        inf_module = RaggedOpsBuilder().load()
        self.kernel = inf_module.moe_scatter

    def __call__(self, moe_input: torch.Tensor, expert_cumsum: torch.Tensor, mapped_slots: torch.Tensor,
                 activations: torch.Tensor, expert_counts: torch.Tensor, assignments: torch.Tensor,
                 offsets: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Scatters the hidden states such that the token stride for each expert's input is contiguous.

        Arguments:
            moe_input (torch.Tensor): The direct input for the MoE GEMM of shape [n_tokens * n_top_k, hidden_size].
            expert_cumsum (torch.Tensor): The cumulative sum of the expert counts of shape [n_experts].
            mapped_slots (torch.Tensor): The index of the token in the expert's input of shape [n_tokens, n_top_k].
            hidden_states (torch.Tensor): The hidden states of shape [n_tokens, hidden_size].
            expert_counts (torch.Tensor): The number of tokens assigned to each expert of shape [n_experts].
            assignments (torch.Tensor): The expert assignments of shape [n_tokens, n_top_k].
            offsets (torch.Tensor): The offsets into the expert for a given token of shape [n_tokens, n_top_K].

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: The MoE input (with scattered values), the cumsum of the offsets (for the MoE kernels themselves), and the assignments Tensor modified in place to show which row that token was mapped to in the input.
        """
        self.kernel(moe_input, expert_cumsum, mapped_slots, activations, expert_counts, assignments, offsets)
        return moe_input, expert_cumsum, mapped_slots
