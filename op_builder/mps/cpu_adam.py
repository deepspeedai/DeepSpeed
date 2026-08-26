# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import os
import subprocess

from .builder import MPSOpBuilder


class CPUAdamBuilder(MPSOpBuilder):
    """Builds the C++ CPU Adam kernel on Apple Silicon for ZeRO-Offload.

    Unified memory means the optimizer step can run on the CPU cores without copying parameters
    off the GPU, so this is the offload path on Macs until a Metal Adam kernel exists.
    """
    BUILD_VAR = "DS_BUILD_CPU_ADAM"
    NAME = "cpu_adam"

    def __init__(self):
        super().__init__(name=self.NAME)

    def absolute_name(self):
        return f'deepspeed.ops.adam.{self.NAME}_op'

    def sources(self):
        return ['csrc/adam/cpu_adam.cpp', 'csrc/adam/cpu_adam_impl.cpp']

    def include_paths(self):
        return ['csrc/includes']

    def builder(self):
        from torch.utils.cpp_extension import CppExtension
        include_dirs = [os.path.abspath(path) for path in self.strip_empty_entries(self.include_paths())]
        return CppExtension(name=self.absolute_name(),
                            sources=self.strip_empty_entries(self.sources()),
                            include_dirs=include_dirs,
                            extra_compile_args={'cxx': self.strip_empty_entries(self.cxx_args())},
                            extra_link_args=self.strip_empty_entries(self.extra_ldflags()))

    def _libomp_prefix(self):
        # Apple clang ships without OpenMP; Homebrew's libomp provides it when installed.
        try:
            return subprocess.check_output(['brew', '--prefix', 'libomp'], stderr=subprocess.DEVNULL).decode().strip()
        except (OSError, subprocess.CalledProcessError):
            return None

    def cxx_args(self):
        args = ['-O3', '-std=c++17', '-g', '-Wno-reorder', '-D__SCALAR__']
        libomp = self._libomp_prefix()
        if libomp is not None:
            args += ['-Xpreprocessor', '-fopenmp', f'-I{libomp}/include']
        return args

    def extra_ldflags(self):
        libomp = self._libomp_prefix()
        if libomp is None:
            return []
        return [f'-L{libomp}/lib', '-lomp']
