# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
import os

from .builder import CUDAOpBuilder, installed_cuda_version


class InferenceCutlassBuilder(CUDAOpBuilder):
    BUILD_VAR = "DS_BUILD_CUTLASS_OPS"
    NAME = "cutlass_ops"

    def __init__(self, name=None):
        name = self.NAME if name is None else name
        super().__init__(name=name)

    def absolute_name(self):
        return f'deepspeed.inference.v2.kernels.cutlass_ops.{self.NAME}'

    def is_compatible(self, verbose=True):
        try:
            import torch
        except ImportError:
            self.warning("Please install torch if trying to pre-compile inference kernels")
            return False

        cuda_okay = True
        if not self.is_rocm_pytorch() and torch.cuda.is_available():  #ignore-cuda
            sys_cuda_major, _ = installed_cuda_version()
            torch_cuda_major = int(torch.version.cuda.split('.')[0])
            cuda_capability = torch.cuda.get_device_properties(0).major  #ignore-cuda
            if cuda_capability < 6:
                self.warning("NVIDIA Inference is only supported on Pascal and newer architectures")
                cuda_okay = False
            if cuda_capability >= 8:
                if torch_cuda_major < 11 or sys_cuda_major < 11:
                    self.warning("On Ampere and higher architectures please use CUDA 11+")
                    cuda_okay = False
        return super().is_compatible(verbose) and cuda_okay

    def filter_ccs(self, ccs):
        ccs_retained = []
        ccs_pruned = []
        for cc in ccs:
            if int(cc[0]) >= 8:
                # Only support Ampere and newer
                ccs_retained.append(cc)
            else:
                ccs_pruned.append(cc)
        if len(ccs_pruned) > 0:
            self.warning(f"Filtered compute capabilities {ccs_pruned}")
        return ccs_retained

    def get_prefix(self):
        ds_path = self.deepspeed_src_path("deepspeed")
        return "deepspeed" if os.path.isdir(ds_path) else ".."

    def sources(self):
        sources = [
            "inference/v2/kernels/cutlass_ops/cutlass_ops.cpp",
            "inference/v2/kernels/cutlass_ops/mixed_gemm/mixed_gemm.cu",
            "inference/v2/kernels/cutlass_ops/moe_gemm/moe_gemm.cu",
        ]

        prefix = self.get_prefix()
        sources = [os.path.join(prefix, src) for src in sources]
        return sources

    def extra_ldflags(self):
        import dskernels
        lib_path = dskernels.library_path()
        prefix = self.get_prefix()
        lib_path = os.path.join(prefix, lib_path)
        lib_path = self.deepspeed_src_path(lib_path)

        args = [f'-L{lib_path}', '-ldeepspeedft']
        if self.jit_load:
            args.append(f'-Wl,-rpath,{lib_path}')
        return args

    def include_paths(self):
        sources = [
            'inference/v2/kernels/includes',
            'inference/v2/kernels/cutlass_ops/mixed_gemm',
            'inference/v2/kernels/cutlass_ops/moe_gemm',
            'inference/v2/kernels/cutlass_ops/shared_resources/',
        ]

        prefix = self.get_prefix()
        sources = [os.path.join(prefix, src) for src in sources]
        return sources
