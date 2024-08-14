# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

from .builder import OpBuilder

try:
    from packaging import version as pkg_version
except ImportError:
    pkg_version = None


class SparseAttnBuilder(OpBuilder):
    BUILD_VAR = "DS_BUILD_SPARSE_ATTN"
    NAME = "sparse_attn"

    def __init__(self):
        super().__init__(name=self.NAME)

    def absolute_name(self):
        return f'deepspeed.ops.sparse_attention.{self.NAME}_op'

    def sources(self):
        return ['csrc/sparse_attention/utils.cpp']

    def cxx_args(self):
        return ['-O2', '-fopenmp']

    def is_compatible(self, verbose=False):
        # Check to see if llvm and cmake are installed since they are dependencies
        #required_commands = ['llvm-config|llvm-config-9', 'cmake']
        #command_status = list(map(self.command_exists, required_commands))
        #deps_compatible = all(command_status)

        if self.is_rocm_pytorch():
            if verbose:
                self.warning(f'{self.NAME} is not compatible with ROCM')
            return False

        try:
            import torch
        except ImportError:
            if verbose:
                self.warning(f"unable to import torch, please install it first")
            return False

        # torch-cpu will not have a cuda version
        if torch.version.cuda is None:
            cuda_compatible = False
            if verbose:
                self.warning(f"{self.NAME} cuda is not available from torch")
        else:
            major, minor = torch.version.cuda.split('.')[:2]
            cuda_compatible = (int(major) == 10 and int(minor) >= 1) or (int(major) >= 11)
            if not cuda_compatible:
                if verbose:
                    self.warning(f"{self.NAME} requires CUDA version 10.1+")

        TORCH_MAJOR = int(torch.__version__.split('.')[0])
        TORCH_MINOR = int(torch.__version__.split('.')[1])
        torch_compatible = (TORCH_MAJOR == 1 and TORCH_MINOR >= 5)
        if not torch_compatible:
            if verbose:
                self.warning(
                    f'{self.NAME} requires a torch version >= 1.5 and < 2.0 but detected {TORCH_MAJOR}.{TORCH_MINOR}')

        try:
            import triton
        except ImportError:
            # auto-install of triton is broken on some systems, reverting to manual install for now
            # see this issue: https://github.com/microsoft/DeepSpeed/issues/1710
            if verbose:
                self.warning(f"please install triton==1.0.0 if you want to use sparse attention")
            return False

        if pkg_version:
            installed_triton = pkg_version.parse(triton.__version__)
            triton_mismatch = installed_triton != pkg_version.parse("1.0.0")
        else:
            installed_triton = triton.__version__
            triton_mismatch = installed_triton != "1.0.0"

        if triton_mismatch:
            if verbose:
                self.warning(
                    f"using untested triton version ({installed_triton}), only 1.0.0 is known to be compatible")
            return False

        return super().is_compatible(verbose) and torch_compatible and cuda_compatible
