# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import torch

from ... import DSKernelBase
from deepspeed.ops.op_builder import RaggedOpsBuilder
from ....inference_utils import elem_size
from ....ragged import RaggedBatchWrapper


class RaggedLogitsGather(DSKernelBase):
    """
    CUDA Kernel implementation for gather the hidden states of the final token
    of each sequence. This is used to reduce the cost of the performing the unembedding.
    """

    supported_dtypes = [torch.float16, torch.bfloat16, torch.float32]

    def __init__(self, model_dim: int, fp_dtype: torch.dtype):
        """
        Parameters:
            fp_dtype (torch.dtype): Data type for the input/output. Supported values
                are torch.float16, torch.bfloat16, and torch.float32.
        """
        if fp_dtype not in RaggedLogitsGather.supported_dtypes:
            raise ValueError("Unsupported data type: {}, supported_dtypes are {}".format(
                fp_dtype, RaggedLogitsGather.supported_dtypes))

        if elem_size(fp_dtype) * model_dim % 16 != 0:
            raise ValueError("Embedding dimension must be aligned to 16 bytes, got {}".format(model_dim))

        inf_module = RaggedOpsBuilder().load()
        self.kernel = inf_module.gather_for_logits

    def __call__(self, final_token_activations: torch.Tensor, all_activations: torch.Tensor,
                 ragged_wrapper: RaggedBatchWrapper) -> torch.Tensor:
        """
        Gather the hidden states of the final token of each sequence from `all_activations` into
        `final_token_activations`.

        Args:
            final_token_activations (torch.Tensor): Output tensor of shape [num_seqs, model_dim]
            all_activations (torch.Tensor): Input tensor of shape [num_tokens, model_dim]
            ragged_wrapper (RaggedBatchWrapper): Wrapper for the ragged batch.
        """

        self.kernel(final_token_activations, all_activations, ragged_wrapper.batch_metadata_buffer(),
                    ragged_wrapper.inflight_seq_descriptors())
        return final_token_activations
