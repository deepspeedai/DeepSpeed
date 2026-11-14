# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

from .builder import CUDAOpBuilder


class DeepCompileBuilder(CUDAOpBuilder):
    BUILD_VAR = "DS_BUILD_DEEP_COMPILE"
    NAME = "dc"

    def __init__(self):
        super().__init__(name=self.NAME)

    def absolute_name(self):
        return f'deepspeed.ops.{self.NAME}_op'

    def sources(self):
        return [
            'csrc/compile/deepcompile.cpp', 'csrc/compile/init.cpp', 'csrc/compile/z1.cpp', 'csrc/compile/z2.cpp',
            'csrc/compile/z3.cpp', 'csrc/compile/util.cpp'
        ]

    def libraries_args(self):
        args = super().libraries_args()
        return args

    @staticmethod
    def _nccl_wheel_include_dir():
        # A pip-installed torch uses NCCL from the nvidia-nccl wheel, which puts nccl.h in
        # site-packages/nvidia/nccl/include, outside the compiler's search path (#7468).
        import importlib.metadata
        import os
        import sys
        try:
            torch_requirements = importlib.metadata.requires("torch") or []
        except importlib.metadata.PackageNotFoundError:
            return None
        if not any(requirement.startswith("nvidia-nccl") for requirement in torch_requirements):
            return None
        # Search sys.path the same way torch searches it for the NCCL library.
        for path in sys.path:
            include_dir = os.path.abspath(os.path.join(path, "nvidia", "nccl", "include"))
            if os.path.isfile(os.path.join(include_dir, "nccl.h")):
                return include_dir
        return None

    def include_paths(self):
        import os
        import torch
        if self.build_for_cpu:
            CUDA_INCLUDE = []
        elif not self.is_rocm_pytorch():
            CUDA_INCLUDE = [os.path.join(torch.utils.cpp_extension.CUDA_HOME, "include")]
            nccl_include = self._nccl_wheel_include_dir()
            if nccl_include:
                CUDA_INCLUDE.append(nccl_include)
        else:
            CUDA_INCLUDE = [
                os.path.join(torch.utils.cpp_extension.ROCM_HOME, "include"),
            ]
        return ['csrc/includes', 'csrc/compile'] + CUDA_INCLUDE
