# Copyright (c) 2023 Habana Labs, Ltd. an Intel Company
# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0
import importlib

# DeepSpeed Team

try:
    # is op_builder from deepspeed or a 3p version? this should only succeed if it's deepspeed
    # if successful this also means we're doing a local install and not JIT compile path
    from op_builder import __deepspeed__  # noqa: F401
    from op_builder.builder import OpBuilder
except ImportError:
    from deepspeed.ops.op_builder.builder import OpBuilder


class InferenceBuilder(OpBuilder):
    BUILD_VAR = "DS_BUILD_TRANSFORMER_INFERENCE"
    NAME = "transformer_inference"

    def __init__(self, name=None):
        name = self.NAME if name is None else name
        super().__init__(name=self.NAME)

    def absolute_name(self):
        return f"deepspeed.ops.transformer.inference.{self.NAME}_op"

    def sources(self):
        return []

    def load(self, verbose=True):
        if self.name in __class__._loaded_ops:
            return __class__._loaded_ops[self.name]

        from deepspeed.git_version_info import installed_ops  # noqa: F401
        if installed_ops.get(self.name, False):
            op_module = importlib.import_module(self.absolute_name())
            __class__._loaded_ops[self.name] = op_module
            return op_module
