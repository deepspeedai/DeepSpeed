# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

from abc import abstractmethod
from typing import Any, Dict, Optional, Tuple, Type

import torch

from deepspeed.runtime.config_utils import DeepSpeedConfigModel
from ..ds_module import DSModuleBase
from ..configs.norm_config import DSNormConfig
from ..module_registry import DSModuleRegistryBase
from ...inference_parameter import InferenceParameter


class DSPostNormBase(DSModuleBase):
    """
    Base MixIn for all Post-Normalization modules. The interface represented by this
    module is:

    residual, hidden_out = norm(residual + hidden_in)

    If residual and hidden_out are the same data type, then they may alias each other.
    Furthermore, residual should be updated in-place.
    """

    @staticmethod
    def config_class() -> Type[DeepSpeedConfigModel]:
        return DSNormConfig

    def __init__(self, config: DSNormConfig, implementation_config: Dict[str, Any]) -> None:
        super().__init__(config, implementation_config)

    @abstractmethod
    def transform_param(self, param: torch.Tensor) -> InferenceParameter:
        """
        Transform a gamma/beta parameter. It is assumed that both transformations are
        the same.

        Parameters:
            param (torch.Tensor): Gamma or beta parameter.
        """
        ...

    def forward(self,
                residual: torch.Tensor,
                hidden_states: torch.Tensor,
                gamma: torch.Tensor,
                beta: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters:
            residual (torch.Tensor): Residual tensor.
            hidden_states (torch.Tensor): Hidden states tensor.

        Returns:
            (torch.Tensor, torch.Tensor): Tuple of residual and hidden states.
                Hidden states may alias with residual.
        """
        raise NotImplementedError()


class DSPostNormRegistry(DSModuleRegistryBase):
    registry: Dict = {}

    @staticmethod
    def associated_class() -> Type[DSModuleBase]:
        return DSPostNormBase
