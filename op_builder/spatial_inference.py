# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

from .builder import CUDAOpBuilder


class SpatialInferenceBuilder(CUDAOpBuilder):
    BUILD_VAR = "DS_BUILD_SPATIAL_INFERENCE"
    NAME = "spatial_inference"
    MIN_COMPUTE_CAPABILITY = 0  # Any GPU; setting it keeps is_compatible()'s torch and CUDA toolkit checks.

    def __init__(self, name=None):
        name = self.NAME if name is None else name
        super().__init__(name=name)

    def absolute_name(self):
        return f'deepspeed.ops.spatial.{self.NAME}_op'

    def sources(self):
        return [
            'csrc/spatial/csrc/opt_bias_add.cu',
            'csrc/spatial/csrc/pt_binding.cpp',
        ]

    def include_paths(self):
        return ['csrc/spatial/includes', 'csrc/includes']
