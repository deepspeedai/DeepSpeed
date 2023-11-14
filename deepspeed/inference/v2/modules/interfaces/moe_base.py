# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

from abc import abstractmethod
from typing import Any, Dict, Optional, Type

import torch

from deepspeed.runtime.config_utils import DeepSpeedConfigModel
from ..ds_module import DSModuleBase
from ..module_registry import DSModuleRegistryBase
from ..configs import DSMoEConfig
from ...inference_parameter import InferenceParameter


class DSMoEBase(DSModuleBase):
    """
    Base mixing for MoE modules. The interface represented by this module is:

    expert_assignments = gate(hidden_states)
    intermediate = ragged_linear(hidden_states, expert_assignments)
    output = ragged_linear(intermediate, expert_assignments)
    """

    @staticmethod
    def config_class() -> Type[DeepSpeedConfigModel]:
        return DSMoEConfig

    def __init__(self, config: DSMoEConfig, implementation_config: Dict[str, Any]) -> None:
        super().__init__(config, implementation_config)

    @abstractmethod
    def transform_gate_param(self, param: torch.Tensor) -> InferenceParameter:
        """
        Perform any necessary transformations of the gate parameter.

        Args:
            param (torch.Tensor): gate_w (shape: [num_experts, model_dim])
        """
        ...

    @abstractmethod
    def transform_moe_mlp_1_param(self, param: torch.Tensor) -> InferenceParameter:
        """
        Perform any necessary transformations of the parameter. The specific component
        being transformed should be inferred from the shape of the parameter.

        Args:
            param (torch.Tensor): One of either mlp_1_w, mlp_1_b
        """
        ...

    @abstractmethod
    def transform_moe_mlp_2_param(self, param: torch.Tensor) -> InferenceParameter:
        """
        Perform any necessary transformations of the parameter. The specified component being
        transformed should be inferred from the shape of the parameter. This interface is
        separate from transform_moe_1_param because the two components may have identical
        shapes.

        Args:
            param (torch.Tensor): One of either mlp_2_w or mlp_2_b
        """
        ...

    def forward(self,
                hidden_states: torch.Tensor,
                gate_w: torch.Tensor,
                mlp_1_w: torch.Tensor,
                mlp_2_w: torch.Tensor,
                mlp_1_b: Optional[torch.Tensor] = None,
                mlp_2_b: Optional[torch.Tensor] = None) -> torch.Tensor:
        raise NotImplementedError()

    @property
    @abstractmethod
    def output(self) -> torch.Tensor:
        """
        Returns the pre-allocated, padded output Tensor.
        """
        ...


class DSMoERegistry(DSModuleRegistryBase):
    registry: Dict = {}

    @staticmethod
    def associated_class() -> Type[DSModuleBase]:
        return DSMoEBase
