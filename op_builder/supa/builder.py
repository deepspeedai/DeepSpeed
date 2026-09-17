# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team

try:
    from op_builder import __deepspeed__  # noqa: F401 # type: ignore
    from op_builder.builder import OpBuilder
except ImportError:
    from deepspeed.ops.op_builder.builder import OpBuilder


class SUPAOpBuilder(OpBuilder):
    """Base class for SUPA operator builders."""

    def builder(self):
        from torch.utils.cpp_extension import CppExtension as ExtensionBuilder

        compile_args = {'cxx': self.strip_empty_entries(self.cxx_args())}

        cpp_ext = ExtensionBuilder(name=self.absolute_name(),
                                   sources=self.strip_empty_entries(self.sources()),
                                   include_dirs=self.strip_empty_entries(self.include_paths()),
                                   libraries=self.strip_empty_entries(self.libraries_args()),
                                   extra_compile_args=compile_args,
                                   extra_link_args=self.strip_empty_entries(self.extra_ldflags()))

        return cpp_ext

    def cxx_args(self):
        CPU_ARCH = self.cpu_arch()
        SIMD_WIDTH = self.simd_width()
        return ['-O3', '-std=c++17', '-g', '-Wno-reorder', '-fopenmp', CPU_ARCH, SIMD_WIDTH]

    def libraries_args(self):
        return []
