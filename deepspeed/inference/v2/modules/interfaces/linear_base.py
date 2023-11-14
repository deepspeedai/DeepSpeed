# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

from abc import abstractmethod
from typing import Any, Dict, Optional, Type

import torch

from deepspeed.runtime.config_utils import DeepSpeedConfigModel
from ..ds_module import DSModuleBase
from ..module_registry import DSModuleRegistryBase
from ..configs import DSLinearConfig
from ...inference_parameter import InferenceParameter


class DSLinearBase(DSModuleBase):
    """
    Base mixin for all Linear modules. The interface represented by this module
    is:

    hidden_out = activation(hidden_in * weight + bias)

    The format and dtype of the weight and bias tensors are not defined and implementations
    may compress as necessary. Must support a bias.
    """

    @staticmethod
    def config_class() -> Type[DeepSpeedConfigModel]:
        return DSLinearConfig

    def __init__(self, config: DSLinearConfig, implementation_config: Dict[str, Any]) -> None:
        super().__init__(config, implementation_config)

    @abstractmethod
    def transform_param(self, param: torch.Tensor) -> InferenceParameter:
        """
        Perform any necessary transformations of the parameters of this module.

        Parameters:
            param (torch.Tensor): Weight or bias tensor.
        """
        ...

    def forward(self, hidden_states: torch.Tensor, w: torch.Tensor, b: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Parameters:
            hidden_states (torch.Tensor): Hidden states tensor. Expected shape is either
                [batch, seq_len, in_channels] or [batch, in_channels].

        Returns:
            torch.Tensor: Output tensor. Tensor should have same number of dimensions as
                input tensor.
        """
        raise NotImplementedError()

    @property
    @abstractmethod
    def output(self) -> torch.Tensor:
        """
        Return the padded, pre-allocated output Tensor.
        """
        ...


class DSLinearRegistry(DSModuleRegistryBase):
    registry: Dict = {}

    @staticmethod
    def associated_class() -> Type[DSModuleBase]:
        return DSLinearBase
