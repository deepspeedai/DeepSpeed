# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import os
from .builder import CUDAOpBuilder
from .cpu.async_io import AsyncIOBuilder


class GDSBuilder(CUDAOpBuilder):
    BUILD_VAR = "DS_BUILD_GDS"
    NAME = "gds"

    def __init__(self):
        super().__init__(name=self.NAME)
        # gds compiles the host-side async_io library into its CUDA extension, so it reuses that op's
        # sources, flags and libaio requirement.
        self.async_io = AsyncIOBuilder()

    def absolute_name(self):
        return f'deepspeed.ops.gds.{self.NAME}_op'

    def lib_sources(self):
        src_list = ['csrc/gds/py_lib/deepspeed_py_gds_handle.cpp', 'csrc/gds/py_lib/deepspeed_gds_op.cpp']
        return self.async_io.lib_sources() + src_list

    def sources(self):
        return self.lib_sources() + ['csrc/gds/py_lib/py_ds_gds.cpp']

    def cxx_args(self):
        return self.async_io.cxx_args() + ['-lcufile']

    def include_paths(self):
        import torch
        CUDA_INCLUDE = [os.path.join(torch.utils.cpp_extension.CUDA_HOME, "include")]
        return ['csrc/aio/py_lib', 'csrc/aio/common', 'csrc/pin_memory'] + CUDA_INCLUDE

    def extra_ldflags(self):
        import torch.utils.cpp_extension
        CUDA_HOME = torch.utils.cpp_extension.CUDA_HOME
        CUDA_LIB64 = os.path.join(CUDA_HOME, "lib64")
        cuda_ldflags = [f'-L{CUDA_HOME}', f'-L{CUDA_LIB64}', '-lcuda', '-lcudart', '-lcufile']
        return self.async_io.extra_ldflags() + cuda_ldflags

    def load(self, verbose=False):
        # The async_io library resolves the shared pin manager from the pin_memory op, so load that first.
        from .pin_memory import PinMemoryBuilder
        PinMemoryBuilder().load(verbose=verbose)
        return super().load(verbose=verbose)

    def is_compatible(self, verbose=False):
        if self.is_rocm_pytorch():
            if verbose:
                self.warning(f'{self.NAME} is not compatible with ROCM')
            return False

        try:
            import torch.utils.cpp_extension
        except ImportError:
            if verbose:
                self.warning("Please install torch if trying to pre-compile GDS")
            return False

        CUDA_HOME = torch.utils.cpp_extension.CUDA_HOME
        if CUDA_HOME is None:
            if verbose:
                self.warning("Please install torch CUDA if trying to pre-compile GDS with CUDA")
            return False

        CUDA_LIB64 = os.path.join(CUDA_HOME, "lib64")
        gds_compatible = self.has_function(funcname="cuFileDriverOpen",
                                           libraries=("cufile", ),
                                           library_dirs=(
                                               CUDA_HOME,
                                               CUDA_LIB64,
                                           ),
                                           verbose=verbose)

        return gds_compatible and self.async_io.is_compatible(verbose) and super().is_compatible(verbose)
