# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import torch

from ...model_implementations.parameter_base import ParameterBase, ParamList
"""
Moe Parameters

These parameters are compatible with any model inheriting from ``DSMoETransformerModelBase``.
"""


class MoEGatingWeightParameter(ParameterBase):
    """
    Gating weight matrix.
    """

    params: torch.Tensor
    """
    Projection matrix from the input activations to the gate logits.
    """

    def finalize(self) -> torch.Tensor:
        return self.inference_model.transform_moe_gate_param(self.params)


class UnfusedMoEMLP1Parameter(ParameterBase):
    """
    This container should be used when the experts are held in separate parameters
    and need to be joined into a single group.
    """

    experts: ParamList("num_experts")  # noqa: F821

    def finalize(self) -> torch.Tensor:
        stacked_experts = torch.stack([p for p in self.experts], dim=0)
        return self.inference_model.transform_moe_mlp_1_param(stacked_experts)


class UnfusedMoEMLP2Parameter(ParameterBase):
    """
    This container should be used when the experts are held in separate parameters
    and need to be joined into a single group.
    """

    experts: ParamList("num_experts")  # noqa: F821

    def finalize(self) -> torch.Tensor:
        stacked_experts = torch.stack([p for p in self.experts], dim=0)
        return self.inference_model.transform_moe_mlp_2_param(stacked_experts)


class UnfusedMoEGatedMLPParameter(ParameterBase):
    """
    MoE Parameter for a gated activation function in which the gating matrix is not
    fused in the same parameter as the non-gating matrix.
    """

    gating_experts: ParamList("num_experts")  # noqa: F821

    up_experts: ParamList("num_experts")  # noqa: F821

    def finalize(self) -> torch.Tensor:
        fused_params = [torch.cat([gate, weight], dim=0) for gate, weight in zip(self.gating_experts, self.up_experts)]
        stacked_params = torch.stack(fused_params, dim=0)
        return self.inference_model.transform_moe_mlp_2_param(stacked_params)
