# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import torch

try:
    from torch.compiler import is_compiling as torch_is_compiling
except ImportError:
    try:
        from torch._dynamo.external_utils import is_compiling as torch_is_compiling
    except ImportError:
        # Torch does not have compiler support
        torch_is_compiling = lambda: False


def is_compile_supported():
    return hasattr(torch, "compiler") and hasattr(torch.nn.Module, "compile")


def disable(func):
    if is_compile_supported():
        return torch.compiler.disable(func)
    return func


def is_compiling():
    return torch_is_compiling()
