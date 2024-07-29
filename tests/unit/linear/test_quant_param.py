# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import pytest
import torch
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
        data = torch.rand(5, 5, device='cpu', dtype=torch.bfloat16)
        qp = QuantizedParameter(data)
        assert qp.device == torch.device('cpu')
        qp = qp.to(get_accelerator().current_device_name())
        assert qp.device == torch.device(device)
        assert qp.dtype == torch.uint8

    def test_hf_clone(self):
        device = get_accelerator().current_device_name()
        data = torch.rand(5, 5, device=device, dtype=torch.bfloat16)

        quantization_config = QuantizationConfig(q_bits=6)
        qp = QuantizedParameter(data, quantization_config=quantization_config)

        # should be able to clone parameter via dict, HF expects this to work
        qp_copy = QuantizedParameter(qp.data, **qp.__dict__)

        assert all(qp.data == qp_copy.data)
        assert qp.quantization_config == qp_copy.quantization_config
