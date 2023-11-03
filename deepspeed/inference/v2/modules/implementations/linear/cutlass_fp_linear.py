# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

from typing import Any, Dict, Optional

import torch

from deepspeed.accelerator import get_accelerator
from ....allocator import empty_from
from ....inference_utils import ActivationType
from ....kernels.core_ops import CUDAGatedActivation

from ...interfaces import DSLinearBase, DSLinearRegistry
from ...configs import DSLinearConfig


@DSLinearRegistry.register_module
class DSCutlassFPLinear(DSLinearBase):
    """
    Linear DSModule based on CUTLASS floating point kernel implementation.
    """

    @staticmethod
    def name():
        return 'cutlass_fp_linear'

    @staticmethod
    def supports_config(config: DSLinearConfig) -> bool:
        if config.input_dtype != config.output_dtype:
            return False

        if config.input_dtype != torch.float16 and config.input_dtype != torch.bfloat16:
            return False

        return True

    def __init__(self, config: DSLinearConfig, implementation_config: Dict[str, Any]) -> None:
        super().__init__(config, implementation_config)

        # TODO: Load kernel

        if config.activation == ActivationType.GEGLU:
            self._geglu = CUDAGatedActivation(config.out_channels, config.output_dtype, ActivationType.GEGLU)
            self._activation_int = torch.empty((config.max_tokens, config.out_channels * 2),
                                               dtype=config.output_dtype,
                                               device=get_accelerator().current_device())

        self._output = torch.empty((config.max_tokens, config.out_channels),
                                   dtype=config.output_dtype,
                                   device=get_accelerator().current_device())

    def transform_param(self, param: torch.Tensor) -> torch.Tensor:
        """
        Converts param to same data type as input and output.

        Parameters:
            param (torch.Tensor): Weight or bias tensor.
        """
        return param.to(self._config.input_dtype)

    def forward(self, hidden_states: torch.Tensor, w: torch.Tensor, b: Optional[torch.Tensor] = None) -> torch.Tensor:

        output = empty_from(self._output, (hidden_states.shape[0], self._config.out_channels))

        if self._config.activation == ActivationType.GEGLU:
            intermediate = empty_from(self._activation_int, (hidden_states.shape[0], self._config.out_channels * 2))
            self._linear_impl(intermediate, hidden_states, w, b)
            self._geglu(output, intermediate)
        else:
            self._linear_impl(output, hidden_states, w, b)

        return output

    @property
    def output(self) -> torch.Tensor:
        """
        Return the padded, pre-allocated output Tensor.
        """
        return self._output
