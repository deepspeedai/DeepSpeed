# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import copy
import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Optional

from deepspeed.accelerator import get_accelerator
from deepspeed.ops.fp_quantizer import Quantizer, FP_Quantize
from .config import QuantizationConfig


class QuantizedParameter(nn.Parameter):
    """
    Quantized parameter class that implements weight quantization. Weights
    are stored in quantized form on GPUs, and can be dequantized on-the-fly when
    needed by the model. The weights are actually quantized during any `.to(device)`.

    Arguments:
        data (Tensor): parameter tensor.
        requires_grad (bool, optional): if the parameter requires gradient. Defaults
            to False and is not supported to be True. Argument provided only for interface
            compatibility with torch.nn.Parameter.
        quantization_config (QuantizationConfig, optional):
        quantizer (Quantizer, optional): Defaults to FP_Quantize but can be any quantizer
            that implements deepspeed.ops.fp_quantizer.Quantizer. This argument is also
            required since the quantizer is stashed in the Parameter itself, some models
            may clone the Parameter by passing an attribute __dict__. For an example, see
            tests/unit/linear/test_quant_param.py::TestQuantParam::test_hf_clone
    """

    def __new__(
        cls,
        data: Optional[torch.Tensor] = None,
        requires_grad: bool = False,  # quantized weights must be frozen
        quantization_config: QuantizationConfig = None,
        quantizer: Quantizer = None,
    ):
        if requires_grad:
            raise ValueError(f"requires_grad=True is not supported with QuantizedParameter")
        if data is None:
            data = torch.empty(0)
        self = torch.Tensor._make_subclass(cls, data, requires_grad)
        self.quantization_config = QuantizationConfig() if quantization_config is None else quantization_config
        if quantizer is not None:
            self.quantizer = quantizer
        else:
            # if FPQuantizerBuilder is not compatible in this env this init will fail
            self.quantizer = FP_Quantize(quantization_config=self.quantization_config)
        self._ensure_quantized(self)
        return self

    def _ensure_quantized(self, tensor: torch.Tensor):
        # If the tensor is on the accelerator and is not quantized, then quantize it in-place.
        if get_accelerator().on_accelerator(tensor) and tensor.dtype != self.quantization_config.q_dtype:
            with get_accelerator().stream(get_accelerator().current_stream(tensor.device)):
                tensor.data = self.quantizer.quantize(tensor.data,
                                                      q_bits=self.quantization_config.q_bits,
                                                      q_mantisa_bits=self.quantization_config.mantissa_bits)
            assert tensor.dtype == self.quantization_config.q_dtype

    def dequantized(self) -> torch.Tensor:
        """
        Return a tensor containing the dequantized weights of this parameter.
        """
        if get_accelerator().on_accelerator(self.data) and self.data.dtype == self.quantization_config.q_dtype:
            with get_accelerator().stream(get_accelerator().current_stream(self.data.device)):
                return self.quantizer.dequantize(self.data,
                                                 q_bits=self.quantization_config.q_bits,
                                                 q_mantisa_bits=self.quantization_config.mantissa_bits)
        return self.data

    def offload(self, revert=False):
        if getattr(self, 'ds_offload', False):
            if revert:
                self.data = self.to(get_accelerator().current_device_name())
            else:
                self.data = self.to('cpu')

    def __getstate__(self):
        state = self.__dict__
        state["data"] = self.data
        state["quantization_config"] = self.quantization_config
        state["requires_grad"] = self.requires_grad
        return state

    def __setstate__(self, state):
        self.quantizer = state["quantizer"]
        self.quantization_config = state["quantization_config"]
        self.data = state["data"]
        self.requires_grad = state["requires_grad"]

    def __deepcopy__(self, memo):
        new_instance = type(self).__new__(type(self))
        state = self.__getstate__()
        new_instance.__setstate__(state)
        new_instance.quantizer = copy.deepcopy(state["quantizer"])
        new_instance.quantization_config = copy.deepcopy(state["quantization_config"])
        new_instance.data = copy.deepcopy(state["data"])
        return new_instance

    def __copy__(self):
        new_instance = type(self).__new__(type(self))
        state = self.__getstate__()
        new_instance.__setstate__(state)
        return new_instance

    def cuda(self, device=None, non_blocking=False):
        device = "cuda" if device is None else device
        self.quantizer.to(device, non_blocking=non_blocking)
        return self.to(device, non_blocking=non_blocking)

    def to(self, *args, **kwargs):
        """
        Move the parameter to the given device. Then, if the device is a cuda device,
        quantize it.
        """
        tensor = super().to(*args, **kwargs)
        self.quantizer.to(*args, **kwargs)
        self._ensure_quantized(tensor)
        return tensor


class QuantizedLinear(nn.Linear):
    """
    Linear layer that implements weight quantization. Parameters
    are stored via `QuantizedParameter` and are dequantized on-the-fly during any
    forward pass.
    """

    def __init__(self,
                 input_dim: int,
                 output_dim: int,
                 bias: bool = False,
                 quantization_config: QuantizationConfig = None,
                 dtype=torch.bfloat16):
        super().__init__(input_dim, output_dim, bias=bias, dtype=dtype)
        assert dtype == torch.bfloat16, "currently only supports bfloat16 dtype"
        self.weight = QuantizedParameter(self.weight.data, quantization_config=quantization_config)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return F.linear(input, self.weight.dequantized(), self.bias)
