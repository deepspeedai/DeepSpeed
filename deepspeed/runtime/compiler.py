# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import torch


def is_compile_supported():
    return hasattr(torch, "compiler") and hasattr(torch.nn.Module, "compile")


def disable(func):
    if is_compile_supported():
        return torch.compiler.disable(func)
    return func
