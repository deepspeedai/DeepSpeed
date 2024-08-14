# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

try:
    from packaging import version as pkg_version
except ImportError:
    pkg_version = None

from .builder import CUDAOpBuilder, installed_cuda_version


class FPQuantizerBuilder(CUDAOpBuilder):
    BUILD_VAR = "DS_BUILD_FP_QUANTIZER"
    NAME = "fp_quantizer"

    def __init__(self, name=None):
        name = self.NAME if name is None else name
        super().__init__(name=name)

    def absolute_name(self):
        return f'deepspeed.ops.fp_quantizer.{self.NAME}_op'

    def is_compatible(self, verbose=False):
        try:
            import torch
        except ImportError:
            if verbose:
                self.warning("Please install torch if trying to pre-compile inference kernels")
            return False

        cuda_okay = True
        if not self.is_rocm_pytorch() and torch.cuda.is_available():  #ignore-cuda
            sys_cuda_major, _ = installed_cuda_version()
            torch_cuda_major = int(torch.version.cuda.split('.')[0])
            cuda_capability = torch.cuda.get_device_properties(0).major  #ignore-cuda
            if cuda_capability < 8:
                if verbose:
                    self.warning("NVIDIA Inference is only supported on Ampere and newer architectures")
                cuda_okay = False
            if cuda_capability >= 8:
                if torch_cuda_major < 11 or sys_cuda_major < 11:
                    if verbose:
                        self.warning("On Ampere and higher architectures please use CUDA 11+")
                    cuda_okay = False

        try:
            import triton
        except ImportError:
            if verbose:
                self.warning(f"please install triton==2.3.0 or 2.3.1 if you want to use the FP Quantizer Kernels")
            return False

        # triton 2.3.0 and 2.3.1 are okay and the only versions released in 2.3.x before 3.x was released
        if pkg_version:
            allowed = pkg_version.parse("2.3")
            installed_triton = pkg_version.parse(triton.__version__)
            triton_mismatch = installed_triton.major != allowed.major or installed_triton.minor != allowed.minor
        else:
            installed_triton = triton.__version__
            major, minor, _ = installed_triton.split(".")
            triton_mismatch = major != "2" or minor != "3"

        if triton_mismatch:
            if verbose:
                self.warning(
                    f"FP Quantizer is using an untested triton version ({installed_triton}), only 2.3.0 and 2.3.1 are known to be compatible with these kernels"
                )
            return False

        return super().is_compatible(verbose) and cuda_okay

    def filter_ccs(self, ccs):
        ccs_retained = []
        ccs_pruned = []
        for cc in ccs:
            if int(cc[0]) >= 8:
                ccs_retained.append(cc)
            else:
                ccs_pruned.append(cc)
        if len(ccs_pruned) > 0:
            self.warning(f"Filtered compute capabilities {ccs_pruned}")
        return ccs_retained

    def sources(self):
        return [
            "csrc/fp_quantizer/fp_quantize.cu",
            "csrc/fp_quantizer/fp_quantize.cpp",
        ]

    def extra_ldflags(self):
        return ['-lcurand']

    def include_paths(self):
        return ['csrc/fp_quantizer/includes', 'csrc/includes']
