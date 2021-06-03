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
        args = [
            '-g',
            '-Wall',
            '-O0',
            '-std=c++14',
            '-shared',
            '-fPIC',
            '-Wno-reorder',
            '-march=native',
            '-fopenmp',
            '-laio',
        ]

        simd_width = self.simd_width()
        if len(simd_width) > 0:
            args.append(simd_width)

        return args

    def extra_ldflags(self):
        return ['-laio']

    def is_compatible(self):
        aio_libraries = ['libaio-dev']
        aio_compatible = self.libraries_installed(aio_libraries)
        if not aio_compatible:
            self.warning(
                f"{self.NAME} requires the libraries: {aio_libraries} but are missing. Can be fixed by: `apt install libaio-dev`."
            )
        return super().is_compatible() and aio_compatible
