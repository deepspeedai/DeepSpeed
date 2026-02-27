# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import torch
from torch.fx import GraphModule
from .passes.sp_compile import apply_autosp

def init_autosp(sp_size=2, dp_size=1):
    def backend_fn(gm: GraphModule, real_inputs):
        apply_autosp(gm, real_inputs, debug=False, sp_size=sp_size, dp_size=dp_size)    
        return torch._inductor.compile(gm, real_inputs)
    return backend_fn
