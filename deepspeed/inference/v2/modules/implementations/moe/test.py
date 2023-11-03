# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import torch
from deepspeed.accelerator import get_accelerator
from deepspeed.inference.modules.interfaces import DSMoERegistry
from deepspeed.inference.modules.configs import DSMoEConfig
from deepspeed.inference.modules.module_registry import ConfigBundle


def run_multi_gemm_with_gating(inputs, gate_weight, moe_weight1, moe_bias1, moe_weight2):
    config = DSMoEConfig(model_dim=4096, intermediate_features=4096, n_experts=64, max_tokens=128)
    moe = DSMoERegistry.instantiate_config(
        ConfigBundle(name='cutlass_multi_gemm_moe',
                     config=config,
                     implementation_config={
                         "weight_dtype": torch.bfloat16,
                         "transpose_weight": True,
                         "min_capacity": 8,
                         "capacity_factor": 1.0
                     }))
    out = moe(inputs, gate_weight, moe_weight1, moe_weight2, moe_bias1)
    return out


a = torch.randn(
    128,
    4096,
).bfloat16().to(get_accelerator().current_device())
weight1 = torch.randn(64, 4096, 4096).bfloat16().to(get_accelerator().current_device())
bias1 = torch.randn(64, 4096).bfloat16().to(get_accelerator().current_device())
weight2 = torch.randn(64, 4096, 4096).bfloat16().to(get_accelerator().current_device())
gate_weight = torch.randn(64, 4096).bfloat16().to(get_accelerator().current_device())

out = run_multi_gemm_with_gating(a, gate_weight, weight1, bias1, weight2)
print(out)
