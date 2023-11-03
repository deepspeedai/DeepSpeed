# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import torch

from ...model_implementations.parameter_base import ParameterBase
"""
MLP Parameter Containers
"""


class MLP1Parameter(ParameterBase):
    """
    First MLP projection weight container. This performs a straight pass-through to the
    model implementation for transformation.
    """
    params: torch.Tensor

    def finalize(self) -> torch.Tensor:
        # NOTE(cmikeh2): If we are gated but not in the format specified below, we should trigger a permutation here.
        # I am not currently aware of any models that use this format (or how we should even detect it; probably should
        # just be a different param entirely, but until then we'll just assume the format is correct).
        return self.inference_model.transform_mlp_1_param(self.params)


class GatedMLPParameter(ParameterBase):
    """
    Gated MLP projection container.
    """

    gate_params: torch.Tensor
    """
    Weight parameter for the gating matrix.
    """

    up_params: torch.Tensor
    """
    For lack of a better name, the non-gating weight parameters.
    """

    def finalize(self) -> torch.Tensor:
        """
        Our gated format (this is different from InferenceV1!) is to have the gate and activated neurons
        interleaved. So if we have 4 output neurons (two effective neurons) with 4 input neurons, the finalized
        parameter will look like:
        [g0_0, g0_1, g0_2, g0_3]
        [a0_0, a0_1, a0_2, a0_3]
        [g1_0, g1_1, g1_2, g1_3]
        [a1_0, a1_1, a1_2, a1_3]

        As a reference, in inference v1, the format is:
        [g0_0, g0_1, g0_2, g0_3]
        [g1_0, g1_1, g1_2, g1_3]
        [a0_0, a0_1, a0_2, a0_3]
        [a1_0, a1_1, a1_2, a1_3]
        """
        assert self.gate_params.shape[0] == self.up_params.shape[
            0], "Gated MLP parameters must have the same number of neurons."
        total_neurons = self.gate_params.shape[0] + self.up_params.shape[0]

        # flip the order if even with the correct tokenizer we get wrong output
        #fused_param = torch.cat([self.up_params, self.gate_params], dim=-1).reshape(total_neurons, -1)
        fused_param = torch.cat([self.gate_params, self.up_params], dim=-1).reshape(total_neurons, -1)
        return self.inference_model.transform_mlp_1_param(fused_param)


class MLP2Parameter(ParameterBase):
    """
    Second MLP projection weight container. This performs a straight pass-through to the
    model implementation for transformation.
    """

    params: torch.Tensor
    """
    Full weight parameter.
    """

    def finalize(self) -> torch.Tensor:
        return self.inference_model.transform_mlp_2_param(self.params)
