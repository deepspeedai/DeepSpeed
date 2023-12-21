# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import torch

from typing import Tuple

from ... import DSKernelBase
from ....inference_utils import DtypeEnum
from ....ragged import RaggedBatchWrapper
from deepspeed.ops.op_builder import RaggedOpsBuilder


class RaggedTopKGating(DSKernelBase):
    """
    CUDA implementation of top-1 gating. This will perform a softmax on the logits,
    and return the scale as well as its idx within that expert's allocation.
    """

    supported_logit_dtypes = [DtypeEnum.fp16, DtypeEnum.bf16, DtypeEnum.fp32]

    def __init__(self, logit_dtype: DtypeEnum) -> None:

        if not isinstance(logit_dtype, DtypeEnum):
            logit_dtype = DtypeEnum(logit_dtype)

        if logit_dtype not in RaggedTopKGating.supported_logit_dtypes:
            raise RuntimeError(f"Unsupported logit dtype {logit_dtype}")

        inf_module = RaggedOpsBuilder().load()
        self.kernel = inf_module.top_k_gating

    def __call__(self, expert_counts: torch.Tensor, scores: torch.Tensor, assignments: torch.Tensor,
                 offsets: torch.Tensor, logits: torch.Tensor,
                 batch: RaggedBatchWrapper) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Perform the ragged top_k_gating.

        Arguments:
            expert_counts (torch.Tensor): Tensor of 0s of shape [n_experts] to be filled with
                number of tokens assigned to each expert. This must be filled with 0s else
                the copy kernel will buffer overflow. In order to minimize the zero-fill cost,
                it is recommended to write to 0 during the MoE output remapping.
            scores (torch.Tensor): Preallocated output of shape [n_tokens, n_top_k] to place expert scaling
                value.
            expert_assignment (torch.Tensor): Preallocated output of shape [n_tokens, n_top_k] to place
                which expert a token has been assigned to.
            expert_offset (torch.Tensor): Preallocated output of shape [n_tokens, n_top_k] to place which
                offset within an experts group a token is.
            logits (torch.Tensor): Raw logits of gating function.
            batch (RaggedBatchWrapper): Batch information for ragged tensor.

        Returns:
            tuple of (expert_counts, scores, expert_assignment, expert_offset)
        """
        self.kernel(expert_counts, scores, assignments, offsets, logits, batch.batch_metadata_buffer())
        return expert_counts, scores, assignments, offsets
