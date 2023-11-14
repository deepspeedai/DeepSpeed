# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

from typing import Any, Dict, Optional, Tuple

import torch

from deepspeed.accelerator import get_accelerator
from ...interfaces import DSPreNormBase, DSPreNormRegistry
from ...configs import DSNormConfig, NormTypeEnum
from ....kernels.core_ops import CUDARMSNorm, CUDARMSPreNorm
from ....allocator import empty_from
from ....inference_parameter import InferenceParameter


@DSPreNormRegistry.register_module
class DSPreRMSCUDAModule(DSPreNormBase):

    @staticmethod
    def name():
        return 'cuda_pre_rms'

    @staticmethod
    def supports_config(config: DSNormConfig):
        type = NormTypeEnum(config.type)
        if type != NormTypeEnum.RMSNorm:
            return False

        if len(set([config.residual_dtype, config.input_dtype, config.output_dtype])) != 1:
            return False

        try:
            # Only need to check one since the support matrix for the two rms kernels is the same
            _ = CUDARMSPreNorm(config.channels, config.residual_dtype)
        except ValueError:
            return False
        return True

    def __init__(self, config: DSNormConfig, implementation_config: Dict[str, Any]):
        super().__init__(config, implementation_config)
        self._fp_rms = CUDARMSNorm(self._config.channels, self._config.residual_dtype, epsilon=self._config.eps)
        self._fp_rms_pre = CUDARMSPreNorm(self._config.channels, self._config.residual_dtype, epsilon=self._config.eps)

        # Buffers for both the hidden and residual outputs
        self._hidden_output = torch.empty((config.max_tokens, config.channels),
                                          dtype=config.output_dtype,
                                          device=get_accelerator().current_device())
        self._residual_output = torch.empty((config.max_tokens, config.channels),
                                            dtype=config.output_dtype,
                                            device=get_accelerator().current_device())

    def transform_param(self, param: torch.Tensor) -> InferenceParameter:
        param = param.to(self._config.input_dtype)
        return InferenceParameter.initialize(param)

    def forward(self,
                residual: torch.Tensor,
                hidden_in: Optional[torch.Tensor],
                gamma: torch.Tensor,
                beta: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Since the CUDA FP only supports all data types being the same, we will alias the residual
        with our output.

        If hidden_in is None, that means we do not need to perform the residual add and will
        only return the hidden output modified.
        """
        assert beta is None, "Beta is not supported for RMSNorm"

        hidden_out = empty_from(self._hidden_output, residual.shape)
        if hidden_in is None:
            self._fp_rms(hidden_out, residual, gamma)
            residual_out = residual
        else:
            residual_out = empty_from(self._residual_output, residual.shape)
            self._fp_rms_pre(residual_out, hidden_out, residual, hidden_in, gamma)
        return residual_out, hidden_out
