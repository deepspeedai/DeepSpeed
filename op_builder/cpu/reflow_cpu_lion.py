# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team

from .reflow import ReflowCPUOp


class ReflowCPULionBuilder(ReflowCPUOp):
    BUILD_VAR = 'DS_BUILD_REFLOW_CPU_LION'
    NAME = 'reflow_cpu_lion'

    def __init__(self):
        super().__init__(name=self.NAME)

    def absolute_name(self):
        return f'deepspeed.ops.lion.{self.NAME}_op'

    def sources(self):
        return ['csrc/lion/reflow_cpu_lion_bindings.cpp', 'csrc/lion/reflow_cpu_lion_impl.cpp']
