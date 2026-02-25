# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import os
import re
from pathlib import Path
from typing import Optional, Tuple

from .builder import CUDAOpBuilder, installed_cuda_version


class EvoformerAttnBuilder(CUDAOpBuilder):
    BUILD_VAR = "DS_BUILD_EVOFORMER_ATTN"
    NAME = "evoformer_attn"

    def __init__(self, name=None):
        name = self.NAME if name is None else name
        super().__init__(name=name)
        self.cutlass_path = os.environ.get("CUTLASS_PATH")
        # Target GPU architecture.
        # Current useful values are: 70, 75, 80.
        # For modern GPUs, 80 is the right value.
        # No specializations of the kernel beyond Ampere are implemented
        # See gemm_kernel_utils.h (also in cutlass example for fused attention) and cutlass/arch/arch.h
        self.gpu_arch = os.environ.get("DS_EVOFORMER_GPU_ARCH")
        self._resolved_gpu_arch = None

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

    @staticmethod
    def _parse_gpu_arch(raw_arch: str) -> Optional[int]:
        token = raw_arch.strip().lower()
        if not token:
            return None

        token = re.sub(r"^sm_?", "", token)
        if "." in token:
            major, minor = token.split(".", maxsplit=1)
            if not (major.isdigit() and minor.isdigit()):
                return None
            return int(major) * 10 + int(minor)

        if not token.isdigit():
            return None

        # Accept single digit forms like "8" and normalize to "80".
        if len(token) == 1:
            return int(token) * 10
        return int(token)

    @staticmethod
    def _parse_cc_token(token: str) -> Tuple[Optional[list], Optional[int]]:
        value = token.strip()
        if not value:
            return None, None

        major, dot, minor = value.partition(".")
        if dot != "." or not major.isdigit():
            return None, None

        minor_value = minor.split("+", maxsplit=1)[0]
        if not minor_value.isdigit():
            return None, None

        return [major, minor], int(major) * 10 + int(minor_value)

    @staticmethod
    def _effective_floor_cc(raw_cc: Optional[int]) -> Optional[int]:
        if raw_cc is None:
            return None
        if raw_cc >= 80:
            return 80
        if raw_cc >= 75:
            return 75
        if raw_cc >= 70:
            return 70
        return None

    def _detect_local_gpu_cc(self) -> Optional[int]:
        try:
            import torch
        except ImportError:
            self.warning("Please install torch if trying to pre-compile kernels")
            return None

        if not torch.cuda.is_available():  #ignore-cuda
            return None

        props = torch.cuda.get_device_properties(0)  #ignore-cuda
        return int(props.major) * 10 + int(props.minor)

    def _resolve_gpu_arch(self) -> Tuple[Optional[int], Optional[int]]:
        if self._resolved_gpu_arch is not None:
            return self._resolved_gpu_arch

        resolved_arch = None
        if self.gpu_arch:
            parsed_arch = self._parse_gpu_arch(self.gpu_arch)
            if parsed_arch is None:
                self.warning(
                    f"Invalid DS_EVOFORMER_GPU_ARCH='{self.gpu_arch}'. Falling back to local CUDA device capability.")
            else:
                resolved_arch = parsed_arch

        if resolved_arch is None:
            resolved_arch = self._detect_local_gpu_cc()

        floor = self._effective_floor_cc(resolved_arch)
        if resolved_arch is not None and floor is None:
            self.warning(f"DS4Sci_EvoformerAttention requires compute capability >= 7.0, got '{resolved_arch}'.")
            resolved_arch = None

        self._resolved_gpu_arch = (resolved_arch, floor)
        return self._resolved_gpu_arch

    def nvcc_args(self):
        args = super().nvcc_args()
        resolved_arch, floor = self._resolve_gpu_arch()
        if floor is None:
            raise RuntimeError(
                "Unable to resolve DS_EVOFORMER_GPU_ARCH for DS4Sci_EvoformerAttention. "
                "Set DS_EVOFORMER_GPU_ARCH to a supported value such as 70, 75, 80, 7.0, 7.5, 8.0, or sm80.")
        if resolved_arch != floor:
            self.warning(
                f"Normalizing DS_EVOFORMER_GPU_ARCH={resolved_arch} to Evoformer kernel family GPU_ARCH={floor}.")
        args.append(f"-DGPU_ARCH={floor}")
        return args

    def filter_ccs(self, ccs):
        _, floor = self._resolve_gpu_arch()

        ccs_retained = []
        ccs_pruned = []
        for cc in ccs:
            parsed_cc, numeric_cc = self._parse_cc_token(cc)
            if parsed_cc is None or numeric_cc is None:
                if cc.strip():
                    ccs_pruned.append(cc.strip())
                continue

            # Evoformer kernels require Volta+.
            if numeric_cc < 70:
                ccs_pruned.append(cc.strip())
                continue

            if floor is not None and numeric_cc < floor:
                ccs_pruned.append(cc.strip())
                continue

            ccs_retained.append(parsed_cc)

        if len(ccs_pruned) > 0:
            if floor is not None:
                self.warning(f"Filtered compute capabilities {ccs_pruned} below Evoformer floor {floor}")
            else:
                self.warning(f"Filtered compute capabilities {ccs_pruned}")

        return ccs_retained

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
