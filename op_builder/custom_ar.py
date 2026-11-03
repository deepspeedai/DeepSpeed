# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team

from .builder import CUDAOpBuilder


class CustomARBuilder(CUDAOpBuilder):
    """One-shot P2P all-reduce for latency-bound TP=2 inference decode."""

    BUILD_VAR = "DS_BUILD_CUSTOM_AR"
    NAME = "custom_ar"

    def __init__(self, name=None):
        super().__init__(name=self.NAME if name is None else name)

    def absolute_name(self):
        return f'deepspeed.ops.{self.NAME}_op'

    def sources(self):
        return ['csrc/inference/custom_ar.cu']
