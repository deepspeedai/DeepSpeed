# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team

from .builder import KLXPUOpBuilder


class UtilsBuilder(KLXPUOpBuilder):
    BUILD_VAR = "DS_BUILD_UTILS"
    NAME = "utils"

    def __init__(self):
        super().__init__(name=self.NAME)

    def absolute_name(self):
        return f'deepspeed.ops.{self.NAME}_op'

    def sources(self):
        return ['csrc/utils/tensor_cast.cpp', 'csrc/utils/py_ds_utils.cpp']

    def include_paths(self):
        return ['csrc/utils', 'csrc/includes']
