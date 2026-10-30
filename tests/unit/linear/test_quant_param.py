# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import pytest
import torch
import torch.nn.functional as F
import deepspeed

from deepspeed.accelerator import get_accelerator
from deepspeed.linear.quantization import QuantizedParameter
from deepspeed.linear.config import QuantizationConfig

from deepspeed.ops.op_builder import FPQuantizerBuilder

from unit.common import DistributedTest

if not deepspeed.ops.__compatible_ops__[FPQuantizerBuilder.NAME]:
    pytest.skip("FPQuantizer op is not available on this system", allow_module_level=True)


class TestQuantParam(DistributedTest):
    world_size = 1

    @pytest.mark.parametrize('dtype', [torch.half, torch.float])
    def test_unsupported_dtypes(self, dtype):
        device = get_accelerator().current_device_name()
        data = torch.rand(5, 5, device='cpu', dtype=dtype)
        qp = QuantizedParameter(data)
        with pytest.raises(AssertionError):
            qp.to(device)

    def test_requires_grad(self):
        data = torch.rand(5, 5, dtype=torch.bfloat16)
        with pytest.raises(ValueError):
            QuantizedParameter(data, requires_grad=True)

    def test_move_to_accelerator(self):
        device = get_accelerator().current_device()
        data = torch.rand(4, 4, device='cpu', dtype=torch.bfloat16)
        quantization_config = QuantizationConfig(group_size=8)
        quantization_config.q_dtype = FPQuantizerBuilder.get_default_quant_dtype()
        qp = QuantizedParameter(data, quantization_config=quantization_config)
        assert qp.device == torch.device('cpu')
        qp = qp.to(get_accelerator().current_device_name())
        assert qp.device == torch.device(device)
        assert qp.dtype == quantization_config.q_dtype

    def test_hf_clone(self):
        device = get_accelerator().current_device_name()
        data = torch.rand(4, 4, device=device, dtype=torch.bfloat16)

        quantization_config = QuantizationConfig(q_bits=6, group_size=8)
        qp = QuantizedParameter(data, quantization_config=quantization_config)

        # should be able to clone parameter via dict, HF expects this to work
        qp_copy = QuantizedParameter(qp.data, **qp.__dict__)

        assert all(qp.data == qp_copy.data)
        assert qp.quantization_config == qp_copy.quantization_config

    @pytest.mark.parametrize("q_bits", [6, 8, 12])
    def test_small_parameter_model_output(self, q_bits):
        device = get_accelerator().current_device_name()
        weight = torch.tensor([[1, 1, 1, 1], [-1, -1, -1, -1], [1, -1, 1, -1], [0, 0, 0, 0]], dtype=torch.bfloat16)
        input = torch.tensor([[1, 2, 3, 4]], dtype=torch.bfloat16, device=device)
        expected = F.linear(input, weight.to(device))
        quantization_config = QuantizationConfig(q_bits=q_bits, group_size=8)
        quantization_config.q_dtype = FPQuantizerBuilder.get_default_quant_dtype()

        quantized_weight = QuantizedParameter(weight, quantization_config=quantization_config).to(device)
        actual = F.linear(input, quantized_weight.dequantized())

        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    @pytest.mark.parametrize("shape", [(1, 4), (5, 5)])
    def test_rejects_incomplete_quantization_group(self, shape):
        device = get_accelerator().current_device_name()
        data = torch.rand(shape, dtype=torch.bfloat16)
        quantization_config = QuantizationConfig(group_size=8)
        quantization_config.q_dtype = FPQuantizerBuilder.get_default_quant_dtype()
        quantized_parameter = QuantizedParameter(data, quantization_config=quantization_config)

        with pytest.raises(ValueError, match="must be divisible by group_size"):
            quantized_parameter.to(device)
