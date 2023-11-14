# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

from .builder import NPUOpBuilder


class NotImplementedBuilder(NPUOpBuilder):
    BUILD_VAR = "DS_BUILD_NOT_IMPLEMENTED"
    NAME = "deepspeed_not_implemented"

    def __init__(self, name=None):
        name = self.NAME if name is None else name
        super().__init__(name=name)

    def absolute_name(self):
        return f'deepspeed.ops.comm.{self.NAME}_op'

    def load(self, verbose=True):
        raise ValueError("This op had not been implemented on NPU backend.")

    def sources(self):
        return []

    def cxx_args(self):
        return []

    def extra_ldflags(self):
        return []

    def include_paths(self):
        return []
