# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team
"""Opt-in deterministic execution policy, independent of numerical compatibility."""

import os

import torch


def configure_determinism(config, param_dict):
    enabled = param_dict.get("deterministic_mode", False)
    if type(enabled) is not bool:
        raise ValueError("deterministic_mode must be a JSON boolean")
    config.deterministic_mode = enabled


def apply_determinism(config):
    if not config.deterministic_mode:
        return
    from deepspeed.accelerator import get_accelerator

    if get_accelerator().device_name() != "cuda":
        raise ValueError("deterministic_mode currently supports CUDA only")
    # Do not set this here: CUDA may already have initialized cuBLAS workspaces.
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") not in (":4096:8", ":16:8"):
        raise ValueError("deterministic_mode requires CUBLAS_WORKSPACE_CONFIG=:4096:8 or :16:8 before CUDA use")
    torch.use_deterministic_algorithms(True, warn_only=False)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
