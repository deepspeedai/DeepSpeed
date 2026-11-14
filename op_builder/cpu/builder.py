# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

try:
    # is op_builder from deepspeed or a 3p version? this should only succeed if it's deepspeed
    # if successful this also means we're doing a local install and not JIT compile path
    from op_builder import __deepspeed__  # noqa: F401 # type: ignore
    from op_builder.builder import OpBuilder
except ImportError:
    from deepspeed.ops.op_builder.builder import OpBuilder


class CPUOpBuilder(OpBuilder):

    def cxx_args(self):
        args = ['-O3', '-g', '-Wno-reorder']
        CPU_ARCH = self.cpu_arch()
        SIMD_WIDTH = self.simd_width()
        args += [CPU_ARCH, '-fopenmp', SIMD_WIDTH]
        return args

    def libraries_args(self):
        return []
