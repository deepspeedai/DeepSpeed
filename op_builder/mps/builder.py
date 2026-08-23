# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

try:
    # is op_builder from deepspeed or a 3p version? this should only succeed if it's deepspeed
    # if successful this also means we're doing a local install and not JIT compile path
    from op_builder import __deepspeed__  # noqa: F401 # type: ignore
    from op_builder.builder import OpBuilder
except ImportError:
    from deepspeed.ops.op_builder.builder import OpBuilder


class MPSOpBuilder(OpBuilder):
    """Base class for ops on Apple Silicon.

    Ops here are currently pure PyTorch (torch.mps) implementations, so there is nothing to compile.
    Metal kernels will plug into the same builder classes later.
    """

    def sources(self):
        return []

    def include_paths(self):
        return []

    def is_compatible(self, verbose=False):
        import torch
        return hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
