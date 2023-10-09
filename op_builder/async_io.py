# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import distutils.spawn
import subprocess

from .builder import OpBuilder


class AsyncIOBuilder(OpBuilder):
    BUILD_VAR = "DS_BUILD_AIO"
    NAME = "async_io"

    def __init__(self):
        super().__init__(name=self.NAME)

    def absolute_name(self):
        return f'deepspeed.ops.aio.{self.NAME}_op'

    def sources(self):
        return [
            'csrc/aio/py_lib/deepspeed_py_copy.cpp', 'csrc/aio/py_lib/py_ds_aio.cpp',
            'csrc/aio/py_lib/deepspeed_py_aio.cpp', 'csrc/aio/py_lib/deepspeed_py_aio_handle.cpp',
            'csrc/aio/py_lib/deepspeed_aio_thread.cpp', 'csrc/aio/common/deepspeed_aio_utils.cpp',
            'csrc/aio/common/deepspeed_aio_common.cpp', 'csrc/aio/common/deepspeed_aio_types.cpp',
            'csrc/aio/py_lib/deepspeed_pin_tensor.cpp'
        ]

    def include_paths(self):
        return ['csrc/aio/py_lib', 'csrc/aio/common']

    def cxx_args(self):
        # -O0 for improved debugging, since performance is bound by I/O
        CPU_ARCH = self.cpu_arch()
        SIMD_WIDTH = self.simd_width()
        import torch  # Keep this import here to avoid errors when building DeepSpeed wheel without torch installed
        TORCH_MAJOR, TORCH_MINOR = map(int, torch.__version__.split('.')[0:2])
        if TORCH_MAJOR >= 2 and TORCH_MINOR >= 1:
            CPP_STD = '-std=c++17'
        else:
            CPP_STD = '-std=c++14'
        return [
            '-g',
            '-Wall',
            '-O0',
            CPP_STD,
            '-shared',
            '-fPIC',
            '-Wno-reorder',
            CPU_ARCH,
            '-fopenmp',
            SIMD_WIDTH,
            '-laio',
        ]

    def extra_ldflags(self):
        return ['-laio']

    def check_for_libaio_pkg(self):
        libs = dict(
            dpkg=["-l", "libaio-dev", "apt"],
            pacman=["-Q", "libaio", "pacman"],
            rpm=["-q", "libaio-devel", "yum"],
        )

        found = False
        for pkgmgr, data in libs.items():
            flag, lib, tool = data
            path = distutils.spawn.find_executable(pkgmgr)
            if path is not None:
                cmd = f"{pkgmgr} {flag} {lib}"
                result = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
                if result.wait() == 0:
                    found = True
                else:
                    self.warning(f"{self.NAME}: please install the {lib} package with {tool}")
                break
        return found

    def is_compatible(self, verbose=True):
        # Check for the existence of libaio by using distutils
        # to compile and link a test program that calls io_submit,
        # which is a function provided by libaio that is used in the async_io op.
        # If needed, one can define -I and -L entries in CFLAGS and LDFLAGS
        # respectively to specify the directories for libaio.h and libaio.so.
        aio_compatible = self.has_function('io_pgetevents', ('aio', ))
        if verbose and not aio_compatible:
            self.warning(f"{self.NAME} requires the dev libaio .so object and headers but these were not found.")

            # Check for the libaio package via known package managers
            # to print suggestions on which package to install.
            self.check_for_libaio_pkg()

            self.warning(
                "If libaio is already installed (perhaps from source), try setting the CFLAGS and LDFLAGS environment variables to where it can be found."
            )
        return super().is_compatible(verbose) and aio_compatible
