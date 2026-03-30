# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import torch
from torch.fx import GraphModule
from .passes.sp_compile import apply_autosp
from .passes.long_context_checkpointing import register_long_context_checkpointing
from .custom_ops.sp_dp_registry import extract_mesh_size
from .custom_ops.sp_compat import _check_autosp_compatibility


def init_autosp(config):
    _check_autosp_compatibility()
    sp_size, dp_size = extract_mesh_size(config._param_dict)
    register_long_context_checkpointing()

    def backend_fn(gm: GraphModule, real_inputs):
        apply_autosp(gm, real_inputs, debug=False, sp_size=sp_size, dp_size=dp_size)
        return torch._inductor.compile(gm, real_inputs)

    return backend_fn
