# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

from .builder import CUDAOpBuilder, installed_cuda_version


class SpatialInferenceBuilder(CUDAOpBuilder):
    BUILD_VAR = "DS_BUILD_SPATIAL_INFERENCE"
    NAME = "spatial_inference"

    def __init__(self, name=None):
        name = self.NAME if name is None else name
        super().__init__(name=name)

    def absolute_name(self):
        return f'deepspeed.ops.spatial.{self.NAME}_op'

    def is_compatible(self, verbose=True):
        try:
            import torch
        except ImportError:
            self.warning("Please install torch if trying to pre-compile inference kernels")
            return False

        cuda_okay = True
        if not self.is_rocm_pytorch() and torch.cuda.is_available():
            sys_cuda_major, _ = installed_cuda_version()
            torch_cuda_major = int(torch.version.cuda.split('.')[0])
            cuda_capability = torch.cuda.get_device_properties(0).major
            if cuda_capability >= 8:
                if torch_cuda_major < 11 or sys_cuda_major < 11:
                    self.warning("On Ampere and higher architectures please use CUDA 11+")
                    cuda_okay = False
        return super().is_compatible(verbose) and cuda_okay

    def sources(self):
        return [
            'csrc/spatial/csrc/opt_bias_add.cu',
            'csrc/spatial/csrc/pt_binding.cpp',
        ]

    def include_paths(self):
        return ['csrc/spatial/includes', 'csrc/includes']
