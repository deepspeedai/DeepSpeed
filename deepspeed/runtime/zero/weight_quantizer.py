# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import math

import torch

from deepspeed.accelerator import get_accelerator


class ZeROWeightQuantizer:
    """Adapter between ZeRO's symmetric int8 weight quantization and the accelerator's quantizer op.

    The grouping is a greedy search for a valid, fast group count, not a search
    for the grouping that minimizes quantization error.
    """

    async_flag = True
    target_group_size = 8000  # empirically good around 4k; keep the target below 8k
    group_size_cache = dict()
    backend_module = None

    def __init__(self) -> None:
        if ZeROWeightQuantizer.backend_module is None:
            from deepspeed.ops.op_builder import QuantizerBuilder
            ZeROWeightQuantizer.backend_module = QuantizerBuilder().load()

    def quantize(self, param, groups=None):
        if groups is None:
            try:
                groups = self.group_size_cache[param.numel()]
            except KeyError:
                groups = math.ceil(param.numel() / self.target_group_size)
                while groups < param.numel():
                    if param.numel() % (8 * groups) == 0:
                        break
                    groups += 1
                while True:
                    if param.numel() % (8 * groups * 2) == 0 and param.numel(
                    ) / groups > self.target_group_size:  # hard limit of 16k group size
                        groups *= 2
                    else:
                        break
                assert param.numel() % (8 * groups) == 0, (
                    f"Quantized weight requires the number of weights to be a multiple of 8; "
                    f"{param.numel()} is not divisible by 8*{groups}")
                assert (param.numel() / groups < 16000), f"{param.numel()} / {groups} is larger than 16k"
                assert param.numel() > groups, (
                    f"Greedy grouping cannot find a group size for input tensor of size {param.numel()}")
                self.group_size_cache[param.numel()] = groups
        # The backend reads fp16; a bf16 input would be reinterpreted. Convert on
        # the way in; dequantize casts back to the caller's dtype.
        param = param.to(get_accelerator().device_name(), dtype=torch.half)
        return self.backend_module.quantize(param, groups, 8, self.backend_module.Symmetric)

    def dequantize(self, quantized_param, scale, dtype=None):
        dequantized = self.backend_module.dequantize(quantized_param, scale, scale.numel(), 8,
                                                     self.backend_module.Symmetric)
        if dtype is not None and dequantized.dtype != dtype:
            dequantized = dequantized.to(dtype)
        return dequantized


# Historic name, kept for import compatibility.
CUDAQuantizer = ZeROWeightQuantizer

_weight_quantizer = None


def get_weight_quantizer():
    """Return the process-wide ZeRO weight quantizer, creating it on first use."""
    global _weight_quantizer
    if _weight_quantizer is None:
        _weight_quantizer = ZeROWeightQuantizer()
    return _weight_quantizer
