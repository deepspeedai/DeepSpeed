# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

from typing import Any, Dict, Optional, Tuple

import torch

from deepspeed.accelerator import get_accelerator
from ...interfaces import DSPreNormBase, DSPreNormRegistry
from ...configs import DSNormConfig, NormTypeEnum
from ....kernels.core_ops.cuda_layer_norm.cuda_pre_ln import CUDAFPPreLN
from ....kernels.core_ops.cuda_layer_norm.cuda_ln import CUDAFPLN
from ....allocator import empty_from
from ....inference_parameter import InferenceParameter


@DSPreNormRegistry.register_module
class DSPreLNCUDAModule(DSPreNormBase):

    @staticmethod
    def name():
        return 'cuda_pre_ln'

    @staticmethod
    def supports_config(config: DSNormConfig):
        type = NormTypeEnum(config.type)
        if type != NormTypeEnum.LayerNorm:
            return False

        if len(set([config.residual_dtype, config.input_dtype, config.output_dtype])) != 1:
            return False

        try:
            _ = CUDAFPPreLN(config.channels, config.residual_dtype)
        except ValueError:
            return False
        return True

    def __init__(self, config: DSNormConfig, implementation_config: Dict[str, Any]):
        super().__init__(config, implementation_config)
        self._fp_pre_ln = CUDAFPPreLN(self._config.channels, self._config.residual_dtype, epsilon=self._config.eps)
        self._fp_ln = CUDAFPLN(self._config.channels, self._config.residual_dtype, epsilon=self._config.eps)

        # Buffers for the hidden output (residual is updated in-place)
        self._hidden_output = torch.empty((config.max_tokens, config.channels),
                                          dtype=config.output_dtype,
                                          device=get_accelerator().current_device())

    def transform_param(self, param: torch.Tensor) -> InferenceParameter:
        param = param.to(self._config.input_dtype)
        return InferenceParameter.initialize(param)

    def forward(self, residual: torch.Tensor, hidden_in: Optional[torch.Tensor], gamma: torch.Tensor,
                beta: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Since the CUDA FP only supports all data types being the same, we will alias the residual
        with our output.

        If hidden_in is None, that means we do not need to perform the residual add and will
        only return the hidden output modified.
        """
        hidden_out = empty_from(self._hidden_output, residual.shape)
        if hidden_in is None:
            self._fp_ln(hidden_out, residual, gamma, beta)
        else:
            self._fp_pre_ln(residual, hidden_out, residual, hidden_in, gamma, beta)
        return residual, hidden_out
