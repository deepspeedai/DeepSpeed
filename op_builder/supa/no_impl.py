# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team

try:
    from op_builder import __deepspeed__  # noqa: F401 # type: ignore
    from op_builder.builder import OpBuilder
except ImportError:
    from deepspeed.ops.op_builder.builder import OpBuilder


class NotImplementedBuilder(OpBuilder):
    """Stub builder for SUPA ops that are not yet implemented."""
    NAME = "not_implemented"
    BUILD_VAR = "DS_BUILD_NOT_IMPLEMENTED"

    def __init__(self, name=None):
        super().__init__(name=name if name is not None else self.NAME)

    def absolute_name(self):
        return f'deepspeed.ops.comm.{self.NAME}_op'

    def sources(self):
        return []

    def cxx_args(self):
        return []

    def extra_ldflags(self):
        return []

    def include_paths(self):
        return []

    def load(self, verbose=True):
        raise NotImplementedError(f"'{self.name}' is not supported on the SUPA accelerator backend.")

    def is_compatible(self, verbose=False):
        return False
