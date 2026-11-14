# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
import os

from .builder import CUDAOpBuilder


class InferenceCutlassBuilder(CUDAOpBuilder):
    BUILD_VAR = "DS_BUILD_CUTLASS_OPS"
    NAME = "cutlass_ops"
    MIN_COMPUTE_CAPABILITY = 8  # Only support Ampere and newer.

    def __init__(self, name=None):
        name = self.NAME if name is None else name
        super().__init__(name=name)

    def absolute_name(self):
        return f'deepspeed.inference.v2.kernels.cutlass_ops.{self.NAME}'

    def get_prefix(self):
        ds_path = self.deepspeed_src_path("deepspeed")
        return "deepspeed" if os.path.isdir(ds_path) else ".."

    def sources(self):
        sources = [
            "inference/v2/kernels/cutlass_ops/cutlass_ops.cpp",
            "inference/v2/kernels/cutlass_ops/mixed_gemm/mixed_gemm.cu",
            "inference/v2/kernels/cutlass_ops/moe_gemm/moe_gemm.cu",
        ]

        prefix = self.get_prefix()
        sources = [os.path.join(prefix, src) for src in sources]
        return sources

    def extra_ldflags(self):
        import dskernels
        lib_path = dskernels.library_path()
        prefix = self.get_prefix()
        lib_path = os.path.join(prefix, lib_path)
        lib_path = self.deepspeed_src_path(lib_path)

        args = [f'-L{lib_path}', '-ldeepspeedft']
        if self.jit_load:
            args.append(f'-Wl,-rpath,{lib_path}')
        return args

    def include_paths(self):
        sources = [
            'inference/v2/kernels/includes',
            'inference/v2/kernels/cutlass_ops/mixed_gemm',
            'inference/v2/kernels/cutlass_ops/moe_gemm',
            'inference/v2/kernels/cutlass_ops/shared_resources/',
        ]

        prefix = self.get_prefix()
        sources = [os.path.join(prefix, src) for src in sources]
        return sources
