# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import torch

from ... import DSKernelBase
from ....inference_utils import ActivationType, DtypeEnum
from deepspeed.ops.op_builder import InferenceCutlassBuilder

from typing import Optional


class MixedMoEGEMM(DSKernelBase):
    """
    CUTLASS implementation of MoE GEMM.
    """

    supported_dtypes = [DtypeEnum.fp16, DtypeEnum.bf16]
    supported_act_fns = [ActivationType.GELU, ActivationType.SILU, ActivationType.RELU, ActivationType.IDENTITY]

    def __init__(self, fp_dtype: DtypeEnum, act_fn: ActivationType, num_bits: int) -> None:

        if not isinstance(fp_dtype, DtypeEnum):
            fp_dtype = DtypeEnum(fp_dtype)

        if fp_dtype not in MixedMoEGEMM.supported_dtypes:
            raise ValueError("Unsupported data type: {}, supported_dtypes are {}".format(
                fp_dtype, MixedMoEGEMM.supported_dtypes))

        if act_fn not in MixedMoEGEMM.supported_act_fns:
            raise ValueError("Unsupported activation function: {}, supported_act_fns are {}".format(
                act_fn, MixedMoEGEMM.supported_act_fns))

        if num_bits != 4 and num_bits != 8:
            raise ValueError("Unsupported num_bits: {}, supported num_bits are 4 and 8".format(num_bits))

        inf_module = InferenceCutlassBuilder().load()
        self.num_bits = num_bits
        self.kernel = inf_module.moe_gemm
        self.act_fn = act_fn

    def __call__(self,
                 ordered_output: torch.Tensor,
                 ordered_input: torch.Tensor,
                 weights: torch.Tensor,
                 scales: torch.Tensor,
                 total_rows_before_expert: torch.Tensor,
                 biases: Optional[torch.Tensor] = None) -> None:
        """
            Performs a MoE GEMM. Note that the stride between token inputs must be even (the distance between byte 1 of token 0 and token 1 must be the same as the distance between byte 1 of token 1 and token 2).

            Arguments:
                ordered_output (torch.Tensor): The output of the MoE GEMM of shape [n_tokens, out_neurons].
                ordered_input (torch.Tensor): The direct input for the MoE GEMM of shape [n_tokens, in_neurons].
                weights (torch.Tensor): The weights of shape [n_experts, in_neurons, out_neurons]. These weights must be contiguous.
                scales (torch.Tensor): The scales of shape [n_experts, out_neurons]. These scales must be contiguous.
                total_rows_before_expert (torch.Tensor): The total number of rows before each expert of shape [n_experts].
                biases (torch.Tensor): The biases of shape [n_experts, out_neurons]. These biases must be contiguous.

            Returns:
                ordered_output
            """
        self.kernel(ordered_output, ordered_input, weights, scales, biases, total_rows_before_expert, self.num_bits,
                    self.act_fn)
        return ordered_output
