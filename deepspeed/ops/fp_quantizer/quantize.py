# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import torch

from deepspeed.ops.op_builder import FPQuantizerBuilder

fp_quant_module = None


class FP_Quantize:

    def __init__(self, group_size=512) -> None:
        global fp_quant_module
        if fp_quant_module is None:
            fp_quant_module = FPQuantizerBuilder().load()

        self.group_size = group_size
        self.orig_dtype = None

    def quantize(self,
                 input,
                 q_bits=8,
                 q_mantisa_bits=3,
                 stochastic_mode=False,
                 return_meta_tensor=False) -> torch.Tensor:
        assert input.dtype == torch.bfloat16, "only support bf16 for now"
        if return_meta_tensor:
            assert q_bits == 8, "meta tensor is only supported with q_bit=8"

        self.orig_dtype = input.dtype
        self.orig_shape = input.shape

        if q_bits == 8:
            pass
        elif q_bits == 12:
            q_mantisa_bits = 4
        elif q_bits == 6:
            q_mantisa_bits = 2
        elif q_bits == 4:
            q_mantisa_bits = 1
        else:
            assert (0), \
                f"Missing {q_bits}-quantization, please add the template arguments for the kernel to support this precision!"

        out = fp_quant_module.quantize(input, self.group_size, stochastic_mode, q_bits, q_mantisa_bits)

        if return_meta_tensor:
            data, scale = out.split(self.group_size, dim=-1)
            return data.contiguous().reshape(input.shape), scale.contiguous()

        return out

    def dequantize(self, input_q, fp_out=None, q_bits=8, q_mantisa_bits=3, scale=None) -> torch.Tensor:
        assert (self.orig_dtype is not None), \
            "[De-quantization Error]: you need to call quantize before dequantizing!"
        fp_out = torch.empty(self.orig_shape, dtype=self.orig_dtype,
                             device=input_q.device) if fp_out is None else fp_out
        if q_bits == 8:
            pass
        elif q_bits == 12:
            q_mantisa_bits = 4
        elif q_bits == 6:
            q_mantisa_bits = 2
        elif q_bits == 4:
            q_mantisa_bits = 1
        else:
            assert (0), \
                f"Missing {q_bits}-dequantization, please add the template arguments for the kernel to support this precision!"

        if scale is not None:
            assert input_q.numel() == fp_out.numel(), \
            f'[De-quantization Error]: quantized data should have the same size as original tensor when scale is not None!'
            input_q = torch.cat([input_q.reshape(-1, self.group_size), scale], dim=-1).contiguous()

        fp_quant_module.dequantize(fp_out, input_q, self.group_size, q_mantisa_bits, q_bits - q_mantisa_bits - 1)
        return fp_out
