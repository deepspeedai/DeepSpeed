# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

from .builder import CUDAOpBuilder, installed_cuda_version
import os
from pathlib import Path


class EvoformerAttnBuilder(CUDAOpBuilder):
    BUILD_VAR = "DS_BUILD_EVOFORMER_ATTN"
    NAME = "evoformer_attn"

    def __init__(self, name=None):
        name = self.NAME if name is None else name
        super().__init__(name=name)
        self.cutlass_path = os.environ.get("CUTLASS_PATH")

    def absolute_name(self):
        return f"deepspeed.ops.{self.NAME}_op"

    def extra_ldflags(self):
        if not self.is_rocm_pytorch():
            return ["-lcurand"]
        else:
            return []

    def sources(self):
        src_dir = "csrc/deepspeed4science/evoformer_attn"
        return [f"{src_dir}/attention.cpp", f"{src_dir}/attention_back.cu", f"{src_dir}/attention_cu.cu"]

    def nvcc_args(self):
        if os.environ.get("DS_EVOFORMER_GPU_ARCH"):
            self.warning("DS_EVOFORMER_GPU_ARCH is deprecated and ignored for Evoformer builds. "
                         "Use TORCH_CUDA_ARCH_LIST to control build targets.")
        return super().nvcc_args()

    def filter_ccs(self, ccs):
        """Keep only Tensor Core capable targets (>= 7.0)."""
        retained = []
        pruned = []
        for cc in [cc.split('.') for cc in ccs]:
            if int(cc[0]) >= 7:
                retained.append(cc)
            else:
                pruned.append(cc)
        if pruned:
            self.warning(f"Evoformer: excluding targets below SM 7.0: {pruned}. Tensor Core required.")
        return retained

    def is_compatible(self, verbose=False):
        try:
            import torch
        except ImportError:
            if verbose:
                self.warning("Please install torch if trying to pre-compile kernels")
            return False

        if self.cutlass_path is None:
            if verbose:
                self.warning("Please specify CUTLASS location directory as environment variable CUTLASS_PATH")
                self.warning(
                    "Possible values are: a path, DS_IGNORE_CUTLASS_DETECTION and DS_USE_CUTLASS_PYTHON_BINDINGS")
            return False

        if self.cutlass_path != "DS_IGNORE_CUTLASS_DETECTION":
            try:
                self.include_paths()
            except (RuntimeError, ImportError):
                return False
            # Check version in case it is a CUTLASS_PATH points to a CUTLASS checkout
            if os.path.exists(f"{self.cutlass_path}/CHANGELOG.md"):
                with open(f"{self.cutlass_path}/CHANGELOG.md", "r") as f:
                    if "3.1.0" not in f.read():
                        if verbose:
                            self.warning("Please use CUTLASS version >= 3.1.0")
                        return False

        # Check CUDA and GPU capabilities
        cuda_okay = True
        if not os.environ.get("DS_IGNORE_CUDA_DETECTION"):
            if not self.is_rocm_pytorch() and torch.cuda.is_available():  #ignore-cuda
                sys_cuda_major, _ = installed_cuda_version()
                torch_cuda_major = int(torch.version.cuda.split(".")[0])
                cuda_capability = torch.cuda.get_device_properties(0).major  #ignore-cuda
                if cuda_capability < 7:
                    if verbose:
                        self.warning("Please use a GPU with compute capability >= 7.0")
                    cuda_okay = False
                if torch_cuda_major < 11 or sys_cuda_major < 11:
                    if verbose:
                        self.warning("Please use CUDA 11+")
                    cuda_okay = False
        return super().is_compatible(verbose) and cuda_okay

    def include_paths(self):
        # Assume the user knows best and CUTLASS location is already setup externally
        if self.cutlass_path == "DS_IGNORE_CUTLASS_DETECTION":
            return []
        # Use header files vendored with deprecated python packages
        if self.cutlass_path == "DS_USE_CUTLASS_PYTHON_BINDINGS":
            try:
                import cutlass_library
                cutlass_path = Path(cutlass_library.__file__).parent / "source"
            except ImportError:
                self.warning("Please pip install nvidia-cutlass (note that this is deprecated and likely outdated)")
                raise
        # Use hardcoded path in CUTLASS_PATH
        else:
            cutlass_path = Path(self.cutlass_path)
        cutlass_path = cutlass_path.resolve()
        if not cutlass_path.is_dir():
            raise RuntimeError(f"CUTLASS_PATH {cutlass_path} does not exist")
        include_dirs = cutlass_path / "include", cutlass_path / "tools" / "util" / "include"
        include_dirs = [str(include_dir) for include_dir in include_dirs if include_dir.is_dir()]
        if not include_dirs:
            raise RuntimeError(f"CUTLASS_PATH {cutlass_path} does not contain any include directories")
        return include_dirs
