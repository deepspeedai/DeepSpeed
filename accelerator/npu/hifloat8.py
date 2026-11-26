# Copyright (c) DeepSpeed Team.
# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team
"""Resolve the optional public TorchAO HiFloat8 APIs on NPU."""

import importlib
from importlib.metadata import PackageNotFoundError, version

from packaging.version import Version


def get_hifloat8_config_class():
    hint = ("Install torchao>=0.18,<0.19, Triton and a compatible Ascend torchao_npu build "
            "providing torchao.hifloat8; keep automatic activation enabled.")
    try:
        installed_version = Version(version("torchao"))
    except PackageNotFoundError as error:
        raise ImportError(f"TorchAO HiFloat8 is unavailable. {hint}") from error
    if not Version("0.18") <= installed_version < Version("0.19"):
        raise RuntimeError(f"Unsupported torchao version {installed_version}. {hint}")
    try:
        return importlib.import_module("torchao.hifloat8").HiFloat8Config
    except (ImportError, AttributeError) as error:
        raise ImportError(f"TorchAO HiFloat8 config is unavailable. {hint}") from error
