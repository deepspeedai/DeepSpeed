# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team

from .builder import CPUOpBuilder

try:
    from op_builder import __deepspeed__  # noqa: F401 # type: ignore
    from op_builder.builder import _cpu_info
except ImportError:
    from deepspeed.ops.op_builder.builder import _cpu_info


class ReflowCPUOp(CPUOpBuilder):
    """Host kernels are currently validated only on x86 CPU and CUDA configurations."""

    def _cpu_flags(self):
        _, flags = _cpu_info()
        return flags

    def simd_width(self):
        width = super().simd_width()
        if width == '-D__AVX512__':
            flags = self._cpu_flags()
            required = {'avx512f', 'avx512bw', 'avx512dq', 'avx512vl'}
            if not required.issubset(flags):
                return '-D__AVX256__' if {'avx2', 'fma', 'f16c'}.issubset(flags) else '-D__SCALAR__'
        return width

    def is_compatible(self, verbose=False):
        try:
            import torch
        except ImportError:
            return False
        try:
            from accelerator import get_accelerator
        except ImportError:
            from deepspeed.accelerator import get_accelerator

        accelerator_name = get_accelerator()._name
        if accelerator_name not in ('cpu', 'cuda') or (accelerator_name == 'cuda' and torch.version.hip is not None):
            if verbose:
                self.warning(f'{self.name} supports only CPU and CUDA accelerator configurations')
            return False
        simd_width = self.simd_width()
        if simd_width not in ('-D__AVX256__', '-D__AVX512__'):
            if verbose:
                self.warning(f'{self.name} requires an x86 CPU with AVX2 or AVX-512')
            return False
        flags = self._cpu_flags()
        required_flags = {'avx2', 'fma', 'f16c'}
        if simd_width == '-D__AVX512__':
            required_flags.update(('avx512f', 'avx512bw', 'avx512dq', 'avx512vl'))
        if not required_flags.issubset(flags):
            if verbose:
                self.warning(f'{self.name} requires CPU flags {sorted(required_flags)}')
            return False
        return super().is_compatible(verbose)

    def include_paths(self):
        return ['csrc/includes']
