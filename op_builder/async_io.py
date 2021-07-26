"""
Copyright 2020 The Microsoft DeepSpeed Team
"""
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
            'csrc/aio/py_lib/deepspeed_py_copy.cpp',
            'csrc/aio/py_lib/py_ds_aio.cpp',
            'csrc/aio/py_lib/deepspeed_py_aio.cpp',
            'csrc/aio/py_lib/deepspeed_py_aio_handle.cpp',
            'csrc/aio/py_lib/deepspeed_aio_thread.cpp',
            'csrc/aio/common/deepspeed_aio_utils.cpp',
            'csrc/aio/common/deepspeed_aio_common.cpp',
            'csrc/aio/common/deepspeed_aio_types.cpp'
        ]

    def include_paths(self):
        return ['csrc/aio/py_lib', 'csrc/aio/common']

    def cxx_args(self):
        # -O0 for improved debugging, since performance is bound by I/O
        CPU_ARCH = self.cpu_arch()
        SIMD_WIDTH = self.simd_width()
        return [
            '-g',
            '-Wall',
            '-O0',
            '-std=c++14',
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

    def is_compatible(self):
        # Check for the existence of libaio by using distutils
        # to compile and link a test program that calls io_submit,
        # which is a function provided by libaio that is used in the async_io op.
        # If needed, one can define -I and -L entries in CFLAGS and LDFLAGS
        # respectively to specify the directories for libaio.h and libaio.so.
        aio_compatible = self.has_function('io_submit', ('aio', ))
        if not aio_compatible:
            self.warning(
                f"{self.NAME} requires libaio but it is missing. Can be fixed by: `apt install libaio-dev`."
            )
        return super().is_compatible() and aio_compatible
