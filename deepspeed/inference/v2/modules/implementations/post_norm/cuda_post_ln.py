# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

from typing import Any, Dict, Tuple

import torch

from deepspeed.accelerator import get_accelerator
from ...interfaces import DSPostNormBase, DSPostNormRegistry
from ...configs import DSNormConfig
from ....kernels.core_ops.cuda_layer_norm.cuda_post_ln import CUDAFPPostLN
from ....allocator import empty_from
from ....inference_parameter import InferenceParameter


@DSPostNormRegistry.register_module
class DSPostLNCUDAModule(DSPostNormBase):

    @staticmethod
    def name():
        return 'cuda_post_ln'

    @staticmethod
    def supports_config(config: DSNormConfig):
        if len(set([config.residual_dtype, config.input_dtype, config.output_dtype])) != 1:
            return False

        try:
            _ = CUDAFPPostLN(config.channels, config.residual_dtype)
        except ValueError:
            return False
        return True

    def __init__(self, config: DSNormConfig, implementation_config: Dict[str, Any]):
        super().__init__(config, implementation_config)
        self._fp_post_ln = CUDAFPPostLN(self._config.channels, self._config.residual_dtype, epsilon=self._config.eps)

        self._output = torch.empty((config.max_tokens, config.channels),
                                   dtype=config.output_dtype,
                                   device=get_accelerator().current_device())

    def transform_param(self, param: torch.Tensor) -> InferenceParameter:
        param = param.to(self._config.input_dtype)
        return InferenceParameter.initialize(param)

    def forward(self, residual: torch.Tensor, hidden_in: torch.Tensor, gamma: torch.Tensor,
                beta: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Since the CUDA FP only supports all data types being the same, we will alias the residual
        with our output.
        """
        self._residual_output = empty_from(self._output, residual.shape)
        self._fp_post_ln(residual, residual, hidden_in, gamma, beta)
        return residual, residual
