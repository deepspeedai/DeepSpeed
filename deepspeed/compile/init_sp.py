# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import torch
from torch.fx import GraphModule
import torch.distributed as dist
from .passes.sp_compile import apply_autosp
from .custom_ops.sp_dp_registry import extract_mesh_size

def init_autosp(compile_config):
    sp_size, dp_size = extract_mesh_size(compile_config)

    def backend_fn(gm: GraphModule, real_inputs):
        apply_autosp(gm, real_inputs, debug=False, sp_size=sp_size, dp_size=dp_size) 
        return torch._inductor.compile(gm, real_inputs)
    return backend_fn
