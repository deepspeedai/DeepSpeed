# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

# Copyright (c) 2023, 2023, Oracle and/or its affiliates.

import os
import torch
import pytest
from unit.common import DistributedTest
import deepspeed
from deepspeed.accelerator import get_accelerator


class TestDequantization(DistributedTest):

    def init(self):
        local_rank = int(os.getenv("LOCAL_RANK", "0"))
        self.device = torch.device(get_accelerator().device_name(local_rank))

        from deepspeed.ops.op_builder import InferenceBuilder
        if not deepspeed.ops.__compatible_ops__[InferenceBuilder.NAME]:
            pytest.skip("InferenceBuilder is not implemented")
        else:
            self.dequantize_func = InferenceBuilder().load().dequantize_fp16

    def run_dequantize_test(self, M, N, num_groups):
        weight = torch.randint(-255, 255, (M, N)).to(dtype=torch.int8, device=self.device)
        scale = torch.rand(num_groups, 1).to(device=self.device)

        weight_deq = (weight.reshape(num_groups, -1) * scale).reshape(M, N).to(torch.float16).contiguous()
        weight_deq_backend = self.dequantize_func(weight, scale, num_groups)

        assert torch.allclose(weight_deq, weight_deq_backend)

    def test_dequantize(self):
        self.init()

        self.run_dequantize_test(14336, 7168, 32)
        self.run_dequantize_test(14336, 1792, 32)
        self.run_dequantize_test(768, 768, 32)
        self.run_dequantize_test(768, 768, 48)
