# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

from .builder import CUDAOpBuilder


class RandomLTDBuilder(CUDAOpBuilder):
    BUILD_VAR = "DS_BUILD_RANDOM_LTD"
    NAME = "random_ltd"

    def __init__(self, name=None):
        name = self.NAME if name is None else name
        super().__init__(name=name)

    def absolute_name(self):
        return f'deepspeed.ops.{self.NAME}_op'

    def extra_ldflags(self):
        if not self.is_rocm_pytorch():
            return ['-lcurand']
        else:
            return []

    def sources(self):
        return [
            'csrc/random_ltd/pt_binding.cpp', 'csrc/random_ltd/gather_scatter.cu',
            'csrc/random_ltd/slice_attn_masks.cu', 'csrc/random_ltd/token_sort.cu'
        ]

    def include_paths(self):
        includes = ['csrc/includes']
        return includes
