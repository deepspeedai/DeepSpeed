# Copyright (c) 2024 Habana Labs, Ltd. an Intel Company
# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import torch
try:
    # is op_builder from deepspeed or a 3p version? this should only succeed if it's deepspeed
    # if successful this also means we're doing a local install and not JIT compile path
    from op_builder import __deepspeed__  # noqa: F401 # type: ignore
    from op_builder.builder import OpBuilder
except ImportError:
    from deepspeed.ops.op_builder.builder import OpBuilder


class FPQuantizerBuilder(OpBuilder):
    BUILD_VAR = "DS_BUILD_FP_QUANTIZER"
    NAME = "fp_quantizer"

    def __init__(self, name=None):
        name = self.NAME if name is None else name
        super().__init__(name=name)

    def absolute_name(self):
        return f'deepspeed.ops.fp_quantizer.{self.NAME}_op'

    def sources(self):
        return []

    def load(self, verbose=True):
        return FPQuantizer

    @staticmethod
    def get_default_quant_dtype():
        return torch.float8_e4m3fn

    @staticmethod
    def get_quant_range(q_bits=None):
        import habana_frameworks.torch.utils.experimental as htexp
        if htexp._get_device_type() == htexp.synDeviceType.synDeviceGaudi2:
            dtype = torch.float8_e4m3fnuz
        else:
            dtype = torch.float8_e4m3fn
        return torch.finfo(dtype).max


class FPQuantizer:
    CUDA_IMPL = False

    @classmethod
    def selective_dequantize(cls, val_q, scales, indexes, group_size, q_mantisa_bits, q_exponent_bits):
        assert False, "Selective dequantize isn't implemented for HPU!"

    @classmethod
    def dequantize(cls, fp_out, input_q, scale, group_size, q_mantisa_bits, q_exponent_bits):
        orig_shape = fp_out.shape
        orig_dtype = fp_out.dtype
        dequant_out = torch.ops.hpu.cast_from_fp8(input_q, (1.0 / scale), orig_dtype).view(orig_shape)
        fp_out.copy_(dequant_out)
        return fp_out

    @classmethod
    def quantize(cls, out, val, scale, group_size, stochastic_rounding, q_bits, q_mantisa_bits):
        assert q_bits == 8, "Quantize on HPU only supports quantization to FP8"
        assert q_mantisa_bits == 3, "Quantize on HPU only supports q_mantissa_bits = 3"
        assert out.dtype.is_floating_point, "Quantization on HPU is only to float dtypes"

        num_groups, group_size = out.shape

        # Reshape the tensor
        val_reshaped = val.view(num_groups, group_size).float()
        # Calculate the scale
        max_vals = val_reshaped.abs().max(dim=1, keepdim=True)[0]
        q_range = torch.finfo(out.dtype).max
        tmp_scale = q_range / max_vals
        scale.copy_(tmp_scale)
        # Copy quantized
        quant, _ = torch.ops.hpu.cast_to_fp8_v2(val_reshaped, scale, stochastic_rounding, dtype=out.dtype)
        out.copy_(quant)

        return out

    @classmethod
    def get_scales(cls, out, num_groups):
        return out
