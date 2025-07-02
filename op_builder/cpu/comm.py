# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import os
from .builder import CPUOpBuilder
import platform


class CCLCommBuilder(CPUOpBuilder):
    BUILD_VAR = "DS_BUILD_CCL_COMM"
    NAME = "deepspeed_ccl_comm"

    def __init__(self, name=None):
        name = self.NAME if name is None else name
        super().__init__(name=name)

    def absolute_name(self):
        return f'deepspeed.ops.comm.{self.NAME}_op'

    def sources(self):
        return ['csrc/cpu/comm/ccl.cpp', 'csrc/cpu/comm/shm.cpp']

    def include_paths(self):
        includes = ['csrc/cpu/includes']
        return includes

    def cxx_args(self):
        return ['-O2', '-fopenmp']

    def is_compatible(self, verbose=False):
        # TODO: add soft compatibility check for private binary release.
        #  a soft check, as in we know it can be trivially changed.
        return super().is_compatible(verbose)

    def extra_ldflags(self):
        ccl_root_path = os.environ.get("CCL_ROOT")
        if ccl_root_path is None:
            raise ValueError(
                "Didn't find CCL_ROOT, install oneCCL from https://github.com/oneapi-src/oneCCL and source its environment variable"
            )
            return []
        else:
            return ['-lccl', f'-L{ccl_root_path}/lib']


class ShareMemCommBuilder(CPUOpBuilder):
    BUILD_VAR = "DS_BUILD_SHM_COMM"
    NAME = "deepspeed_shm_comm"
    machine = platform.machine().lower()

    def __init__(self, name=None):
        name = self.NAME if name is None else name
        super().__init__(name=name)

    def absolute_name(self):
        return f'deepspeed.ops.comm.{self.NAME}_op'

    def sources(self):
        src_files = ['csrc/cpu/comm/shm_interface.cpp']
        if self.machine == 'riscv64':
            src_files.append('csrc/cpu/comm/shm-riscv64.cpp')
        else:
            src_files.append('csrc/cpu/comm/shm.cpp')
        return src_files

    def include_paths(self):
        includes = ['csrc/cpu/includes']
        return includes

    def cxx_args(self):
        arg_list = ['-O2', '-fopenmp']
        if self.machine == 'riscv64':
            arg_list.append('-march=rv64gcv_zvfh')
        return arg_list

    def is_compatible(self, verbose=False):
        # TODO: add soft compatibility check for private binary release.
        #  a soft check, as in we know it can be trivially changed.
        return super().is_compatible(verbose)
