# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

from typing import Any, Dict, Optional

import torch

from deepspeed.accelerator import get_accelerator
from ....allocator import empty_from
from ....inference_utils import is_gated
from ....kernels.core_ops import (
    BlasLibLinear,
    CUDABiasActivation,
    CUDAGatedActivation,
)

from ...interfaces import DSLinearBase, DSLinearRegistry
from ...configs import DSLinearConfig
from ....inference_parameter import InferenceParameter


@DSLinearRegistry.register_module
class BlasFPLinear(DSLinearBase):
    """
    Linear DSModule based on BLAS library and standalone bias + activation kernel implementation.
    """

    @staticmethod
    def name():
        return 'blas_fp_linear'

    @staticmethod
    def supports_config(config: DSLinearConfig) -> bool:
        if config.input_dtype != config.output_dtype:
            return False

        if config.input_dtype != torch.float16 and config.input_dtype != torch.bfloat16:
            return False

        if is_gated(config.activation):
            try:
                _ = CUDAGatedActivation(config.out_channels, config.output_dtype, config.activation)
            except ValueError:
                return False
        else:
            try:
                _ = CUDABiasActivation(config.out_channels, config.output_dtype, config.activation)
            except ValueError:
                return False

        return True

    def __init__(self, config: DSLinearConfig, implementation_config: Dict[str, Any]) -> None:
        super().__init__(config, implementation_config)

        self._linear_impl = BlasLibLinear(self._config.input_dtype)

        if is_gated(config.activation):
            self._is_gated = True
            self._act_fn = CUDAGatedActivation(config.out_channels, config.output_dtype, config.activation)
            self._double_buffer = torch.empty((config.max_tokens, config.out_channels * 2),
                                              dtype=config.output_dtype,
                                              device=get_accelerator().current_device())
        else:
            self._is_gated = False
            self._act_fn = CUDABiasActivation(config.out_channels, config.output_dtype, config.activation)

        self._output = torch.empty((config.max_tokens, config.out_channels),
                                   dtype=config.output_dtype,
                                   device=get_accelerator().current_device())

    def transform_param(self, param: torch.Tensor) -> InferenceParameter:
        """
        Converts param to same data type as input and output.

        Parameters:
            param (torch.Tensor): Weight or bias tensor.
        """
        param = param.to(self._config.output_dtype)
        return InferenceParameter.initialize(param)

    def forward(self, hidden_states: torch.Tensor, w: torch.Tensor, b: Optional[torch.Tensor] = None) -> torch.Tensor:

        output = empty_from(self._output, (hidden_states.shape[0], self._config.out_channels))

        if self._is_gated:
            staging_output = empty_from(self._double_buffer, (hidden_states.shape[0], self._config.out_channels * 2))
            self._linear_impl(staging_output, hidden_states, w)
            self._act_fn(output, staging_output, b)
        else:
            self._linear_impl(output, hidden_states, w)
            self._act_fn(output, b)

        return output

    @property
    def output(self) -> torch.Tensor:
        """
        Return the padded, pre-allocated output Tensor.
        """
        return self._output
