# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import sys

from .builder import OpBuilder
from .pin_memory_load import load_pin_memory_module


class PinMemoryBuilder(OpBuilder):
    BUILD_VAR = "DS_BUILD_PIN_MEMORY"
    NAME = "pin_memory"

    def __init__(self):
        super().__init__(name=self.NAME)

    def absolute_name(self):
        return f'deepspeed.ops.pin_memory.{self.NAME}_op'

    def is_compatible(self, verbose=False):
        # The allocator relies on POSIX mlock/posix_memalign, which are unavailable on Windows.
        if sys.platform == "win32":
            if verbose:
                self.warning(f"{self.NAME} is only supported on POSIX platforms, not Windows.")
            return False
        return super().is_compatible(verbose)

    def sources(self):
        return [
            'csrc/pin_memory/page_alloc.cpp',
            'csrc/pin_memory/deepspeed_pin_tensor.cpp',
            'csrc/pin_memory/py_ds_pin_memory.cpp',
        ]

    def include_paths(self):
        return ['csrc/pin_memory']

    def load(self, verbose=False):
        return load_pin_memory_module(self, verbose=verbose)
