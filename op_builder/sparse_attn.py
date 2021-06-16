"""
Copyright 2020 The Microsoft DeepSpeed Team
"""
import warnings
from .builder import OpBuilder


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

    def is_compatible(self):
        # Check to see if llvm and cmake are installed since they are dependencies
        #required_commands = ['llvm-config|llvm-config-9', 'cmake']
        #command_status = list(map(self.command_exists, required_commands))
        #deps_compatible = all(command_status)

        try:
            import torch
        except ImportError:
            self.warning(f"unable to import torch, please install it first")
            return False

        # torch-cpu will not have a cuda version
        if torch.version.cuda is None:
            cuda_compatible = False
            self.warning(f"{self.NAME} cuda is not available from torch")
        else:
            major, minor = torch.version.cuda.split('.')[:2]
            cuda_compatible = (int(major) == 10
                               and int(minor) >= 1) or (int(major) >= 11)
            if not cuda_compatible:
                self.warning(f"{self.NAME} requires CUDA version 10.1+")

        TORCH_MAJOR = int(torch.__version__.split('.')[0])
        TORCH_MINOR = int(torch.__version__.split('.')[1])
        torch_compatible = TORCH_MAJOR == 1 and TORCH_MINOR >= 5
        if not torch_compatible:
            self.warning(
                f'{self.NAME} requires a torch version >= 1.5 but detected {TORCH_MAJOR}.{TORCH_MINOR}'
            )

        return super().is_compatible() and torch_compatible and cuda_compatible
