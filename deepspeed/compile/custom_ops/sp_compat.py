# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import torch
from packaging.version import Version


def _check_autosp_compatibility():
    # Strip the local version segment (e.g. +cu128) so CUDA builds don't sort
    # above the max bound when using packaging's local-version ordering rules.
    torch_version = Version(torch.__version__.split("+")[0])
    if torch_version < Version("2.9"):
        raise RuntimeError("AutoSP requires PyTorch >= 2.9, found "
                           f"{torch.__version__}.")
