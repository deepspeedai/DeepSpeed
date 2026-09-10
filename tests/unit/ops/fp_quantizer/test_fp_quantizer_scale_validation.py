# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import pytest
import torch

from op_builder.hpu.fp_quantizer import FPQuantizer


def test_dequantize_rejects_non_finite_scale():
    fp_out = torch.zeros(4, dtype=torch.float16)
    input_q = torch.zeros(4, dtype=torch.uint8)
    scale = torch.tensor([float("inf")], dtype=torch.float32)

    with pytest.raises(ValueError, match="dequantize scale must contain finite values"):
        FPQuantizer.dequantize(fp_out, input_q, scale, group_size=1, q_mantisa_bits=3, q_exponent_bits=4)


def test_dequantize_rejects_zero_scale():
    fp_out = torch.zeros(4, dtype=torch.float16)
    input_q = torch.zeros(4, dtype=torch.uint8)
    scale = torch.tensor([0.0], dtype=torch.float32)

    with pytest.raises(ValueError, match="dequantize scale must be non-zero"):
        FPQuantizer.dequantize(fp_out, input_q, scale, group_size=1, q_mantisa_bits=3, q_exponent_bits=4)
