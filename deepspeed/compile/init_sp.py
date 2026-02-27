# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import torch
from torch.fx import GraphModule
from .passes.sp_compile import apply_autosp

def init_autosp(compile_config):
    def backend_fn(gm: GraphModule, real_inputs):
        apply_autosp(gm, real_inputs, debug=False, sp_size=compile_config.sp_size, dp_size=compile_config.dp_size)    
        return torch._inductor.compile(gm, real_inputs)
    return backend_fn
