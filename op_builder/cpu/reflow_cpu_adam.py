# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team

from .reflow import ReflowCPUOp


class ReflowCPUAdamBuilder(ReflowCPUOp):
    BUILD_VAR = 'DS_BUILD_REFLOW_CPU_ADAM'
    NAME = 'reflow_cpu_adam'

    def __init__(self):
        super().__init__(name=self.NAME)

    def absolute_name(self):
        return f'deepspeed.ops.adam.{self.NAME}_op'

    def sources(self):
        return ['csrc/adam/reflow_cpu_adam_bindings.cpp', 'csrc/adam/reflow_cpu_adam_impl.cpp']
