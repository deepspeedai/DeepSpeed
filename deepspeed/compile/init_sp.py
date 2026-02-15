# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import torch
from torch.fx import GraphModule
from .passes.autosp import apply_autosp

def init_ulysses():
    def backend_fn(gm: GraphModule, real_inputs):
        apply_autosp(gm, real_inputs, debug_log=False)    
        return torch._inductor.compile(gm, real_inputs)
    return backend_fn
