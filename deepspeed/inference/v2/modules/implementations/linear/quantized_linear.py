# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

from typing import Any, Dict, Optional

import torch

from deepspeed.accelerator import get_accelerator
from deepspeed.ops.op_builder import InferenceCoreBuilder
from ....allocator import empty_from
from ....inference_utils import is_gated
from ....kernels.core_ops import (
    CUDAWf6Af16Linear,
    CUDABiasActivation,
    CUDAGatedActivation,
)

from ...interfaces import DSLinearBase, DSLinearRegistry
from ...configs import DSLinearConfig
from ....inference_parameter import InferenceParameter


def fp_quantize(input: torch.FloatTensor,
                num_bits: int = 6,
                exp_bits: int = 3,
                min_value: torch.FloatTensor = None,
                max_value: torch.FloatTensor = None,
                group_size: int = -1):
    """
    Args:
        inputs (`torch.FloatTensor`)
            The input which needs to be quantized
        num_bits (int, >=4)
            Number of bits to use for quantization
        exp_bits:
            fp exp_bits
        min_value/max_vlue (torch.FloatTensor)
            Used for static activation quantization
        group_size (int) N
            The quantization block size, each N numbers has its own scaling
            factor and off-site. -1 means use the last dim as the group_size
    Returns:
        quantized_fake_fp6
            The quantized weights, in fp16 format and contains fp6 value.
        scales
            Quantization scales
    """

    try:
        from qtorch.quant import float_quantize
    except ImportError:
        raise ImportError("Please install qtorch to use this function")

    assert (min_value is None and max_value is None) or (min_value is not None and max_value is not None)

    assert input.dtype == torch.float16

    orig_device = input.device
    input = input.to(torch.float32).to(get_accelerator().current_device())
    if num_bits == 6 and exp_bits == 3:  # this is default
        q_range = 28
    else:
        raise NotImplementedError

    man_bits = num_bits - exp_bits - 1
    input_shape = input.shape

    if group_size == -1:
        group_size = input_shape[-1]
    else:
        # Only support per-channel quantization
        raise NotImplementedError
    num_groups = input.numel() // group_size
    input = input.reshape(num_groups, -1)

    if min_value is None:
        max_input = torch.amax(torch.abs(input), dim=-1).view(num_groups, -1)
    else:
        max_input = torch.max(min_value.abs(), max_value)  # .view(-1)
    scales = max_input / q_range  # q_range + 1
    scales[scales == 0] = 1  # avoid zero scales
    scaled_input = input / scales

    quantized_fake_fp6 = float_quantize(scaled_input, exp_bits, man_bits, rounding="nearest")

    quantized_fake_fp6 = quantized_fake_fp6.reshape(input_shape).contiguous().to(torch.float16).to(orig_device)
    scales = scales.to(torch.float16).to(orig_device)
    # Now the dequantized value is quantized_fake_fp6 * scales

    return quantized_fake_fp6, scales


@DSLinearRegistry.register_module
class QuantizedWf6Af16Linear(DSLinearBase):
    """
    Linear DSModule for FP6 weight-only quantization kernel, where weight is FP6
    and activation is FP16.
    """

    @staticmethod
    def name():
        return 'quantized_wf6af16_linear'

    @staticmethod
    def supports_config(config: DSLinearConfig) -> bool:
        if config.input_dtype != config.output_dtype:
            return False

        # As for fp6 data items, they are packed and stored in a set of fp16
        # tensors. E.g., 8 fp6 data items are stored in 3 fp16 tensor.
        if config.input_dtype != torch.float16:
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

        self._linear_impl = CUDAWf6Af16Linear()

        if is_gated(config.activation):
            # In the FP6 kernel implementation, the MatMul is W * A, where W is
            # the weight and A is activation. M is the output channel size.
            self.out_channels = self._config.out_channels * 2
            self.in_channels = self._config.in_channels
            self._is_gated = True
            self._act_fn = CUDAGatedActivation(config.out_channels, config.output_dtype, config.activation)
            self._double_buffer = torch.empty((config.max_tokens, config.out_channels * 2),
                                              dtype=config.output_dtype,
                                              device=get_accelerator().current_device())
        else:
            self.out_channels = self._config.out_channels
            self.in_channels = self._config.in_channels
            self._is_gated = False
            self._act_fn = CUDABiasActivation(config.out_channels, config.output_dtype, config.activation)

        self._output = torch.empty((config.max_tokens, config.out_channels),
                                   dtype=config.output_dtype,
                                   device=get_accelerator().current_device())

        self.inf_module = InferenceCoreBuilder().load()
        self.inf_module.create_handle()
        self.preprocess_weight = self.inf_module.preprocess_weight

        self.quantizer = fp_quantize

    def transform_param(self, param: torch.Tensor) -> InferenceParameter:
        """
        Converts param to same data type as input and output.

        Parameters:
            param (torch.Tensor): Weight or bias tensor.
        """
        # It expects that the quantization scales are store in the attribute `scales`.

        if param.ndim == 1:  # bias, do nothing
            return InferenceParameter.initialize(param)

        quantized_fake_fp6, scales = self.quantizer(param, num_bits=6, exp_bits=3)

        # This is for debugging, will delete before release.
        assert (quantized_fake_fp6.dtype == torch.float16)
        assert quantized_fake_fp6.shape[0] == self.out_channels
        assert scales.numel() == self.out_channels

        weights_2bit, weights_4bit = self.preprocess_weight(quantized_fake_fp6)

        return InferenceParameter.initialize(weights_2bit, weights_4bit=weights_4bit, scales=scales)

    def forward(self, hidden_states: torch.Tensor, w: torch.Tensor, b: Optional[torch.Tensor] = None) -> torch.Tensor:
        weights_2bit = w
        weights_4bit = w.weights_4bit
        scales = w.scales
        output = empty_from(self._output, (hidden_states.shape[0], self._config.out_channels))
        if self._is_gated:
            staging_output = empty_from(self._double_buffer, (hidden_states.shape[0], self.out_channels))
            self._linear_impl(staging_output, hidden_states, weights_2bit, weights_4bit, scales, self.out_channels,
                              hidden_states.shape[0], self.in_channels)
            self._act_fn(output, staging_output, b)
        else:
            self._linear_impl(output, hidden_states, weights_2bit, weights_4bit, scales, self.out_channels,
                              hidden_states.shape[0], self.in_channels)
            self._act_fn(output, b)

        return output

    @property
    def output(self) -> torch.Tensor:
        """
        Return the padded, pre-allocated output Tensor.
        """
        return self._output
