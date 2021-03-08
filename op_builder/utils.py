"""
Copyright 2020 The Microsoft DeepSpeed Team
"""
from .builder import OpBuilder


class UtilsBuilder(OpBuilder):
    BUILD_VAR = "DS_BUILD_UTILS"
    NAME = "utils"

    def __init__(self):
        super().__init__(name=self.NAME)

    def absolute_name(self):
        return f'deepspeed.ops.{self.NAME}_op'

    def sources(self):
        return ['csrc/utils/flatten_unflatten.cpp']
