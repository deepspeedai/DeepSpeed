# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import os

try:
    # is op_builder from deepspeed or a 3p version? this should only succeed if it's deepspeed
    # if successful this also means we're doing a local install and not JIT compile path
    from op_builder import __deepspeed__  # noqa: F401 # type: ignore
    from op_builder.builder import OpBuilder
except ImportError:
    from deepspeed.ops.op_builder.builder import OpBuilder


class MPSOpBuilder(OpBuilder):
    """Base class for ops on Apple Silicon that need no C++ compilation (pure torch.mps implementations)."""

    def sources(self):
        return []

    def include_paths(self):
        return []

    def is_compatible(self, verbose=False):
        import torch
        return hasattr(torch.backends, "mps") and torch.backends.mps.is_available()


class MetalOpBuilder(MPSOpBuilder):
    """Base class for ops implemented as Metal shaders.

    Shaders are compiled at load time through torch.mps.compile_shader, which dispatches them on
    PyTorch's own MPS command stream, so no Xcode toolchain or C++ extension build is needed.
    There is deliberately no torch-extensions style disk cache: runtime compilation of these
    kernels takes milliseconds, and macOS's Metal framework keeps its own per-app cache of
    compiled pipelines. Subclasses list their .metal files in metal_sources() and get back the
    compiled library.
    """
    _libraries = {}

    def metal_sources(self):
        return []

    def is_compatible(self, verbose=False):
        import torch
        return super().is_compatible(verbose) and hasattr(torch.mps, "compile_shader")

    def load_metal_library(self):
        import torch
        if self.name not in MetalOpBuilder._libraries:
            source = ""
            for path in self.metal_sources():
                with open(os.path.join(self.deepspeed_src_path(path))) as shader_file:
                    source += shader_file.read() + "\n"
            MetalOpBuilder._libraries[self.name] = torch.mps.compile_shader(source)
        return MetalOpBuilder._libraries[self.name]
