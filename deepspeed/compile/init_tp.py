# Copyright (c) DeepSpeed Team.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import torch
from torch.fx import GraphModule

from deepspeed.utils.torch import required_torch_version

from .passes.tp_compile import apply_autotp, defer_collectives_to_compiler

AUTOTP_MIN_TORCH_VERSION = 2.6


def _check_autotp_compatibility():
    if not required_torch_version(min_version=AUTOTP_MIN_TORCH_VERSION):
        raise RuntimeError(f"The AutoTP compile pass requires PyTorch >= {AUTOTP_MIN_TORCH_VERSION}, found "
                           f"{torch.__version__}.")


def init_autotp(model):
    """Hand the tensor-parallel collectives of an AutoTP-partitioned model over to the compiler.

    The model is expected to have been partitioned already by the regular AutoTP path, so this only
    suppresses the module-level collectives and returns a backend that emits them as graph nodes.
    """
    _check_autotp_compatibility()
    defer_collectives_to_compiler(model)

    def backend_fn(gm: GraphModule, real_inputs):
        apply_autotp(gm, real_inputs)
        return torch._inductor.compile(gm, real_inputs)

    return backend_fn
