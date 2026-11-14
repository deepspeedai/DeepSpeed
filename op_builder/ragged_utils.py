# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import os

from .builder import CUDAOpBuilder


class RaggedUtilsBuilder(CUDAOpBuilder):
    BUILD_VAR = "DS_BUILD_RAGGED_OPS"
    NAME = "ragged_ops"
    MIN_COMPUTE_CAPABILITY = 6

    def __init__(self, name=None):
        name = self.NAME if name is None else name
        super().__init__(name=name)

    def absolute_name(self):
        return f'deepspeed.inference.v2.{self.NAME}'

    def get_prefix(self):
        ds_path = self.deepspeed_src_path("deepspeed")
        return "deepspeed" if os.path.isdir(ds_path) else ".."

    def sources(self):
        sources = [
            "inference/v2/ragged/csrc/fast_host_buffer.cu",
            "inference/v2/ragged/csrc/ragged_ops.cpp",
        ]

        prefix = self.get_prefix()
        sources = [os.path.join(prefix, src) for src in sources]
        return sources

    def extra_ldflags(self):
        return []

    def include_paths(self):
        include_dirs = ['inference/v2/ragged/includes', 'inference/v2/kernels/includes']
        prefix = self.get_prefix()
        includes = [os.path.join(prefix, include_dir) for include_dir in include_dirs]

        return includes
