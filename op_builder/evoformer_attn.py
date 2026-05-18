# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

from .builder import CUDAOpBuilder, installed_cuda_version
import importlib
import os
from pathlib import Path
import sys


class EvoformerAttnBuilder(CUDAOpBuilder):
    BUILD_VAR = "DS_BUILD_EVOFORMER_ATTN"
    NAME = "evoformer_attn"
    CUTLASS_IGNORE = "DS_IGNORE_CUTLASS_DETECTION"
    CUTLASS_PYTHON_BINDINGS = "DS_USE_CUTLASS_PYTHON_BINDINGS"

    def __init__(self, name=None):
        name = self.NAME if name is None else name
        super().__init__(name=name)
        self.cutlass_path = os.environ.get("CUTLASS_PATH")
        self._resolved_cutlass_path = None

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

        if self.cutlass_path != self.CUTLASS_IGNORE:
            try:
                self.include_paths()
            except (RuntimeError, ImportError) as exc:
                if verbose:
                    self.warning(str(exc))
                return False
            # Check version in case it is a CUTLASS_PATH points to a CUTLASS checkout
            if self._resolved_cutlass_path is not None:
                changelog_path = self._resolved_cutlass_path / "CHANGELOG.md"
            else:
                changelog_path = None
            if changelog_path is not None and changelog_path.exists():
                with open(changelog_path, "r") as f:
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

    @staticmethod
    def _repo_root():
        return Path(__file__).resolve().parents[1]

    @staticmethod
    def _dedupe_paths(paths):
        deduped = []
        seen = set()
        for path in paths:
            path = Path(path).expanduser()
            key = str(path)
            if key not in seen:
                seen.add(key)
                deduped.append(path)
        return deduped

    @staticmethod
    def _env_paths(*names):
        paths = []
        for name in names:
            value = os.environ.get(name)
            if not value:
                continue
            paths.extend(Path(path) for path in value.split(os.pathsep) if path)
        return paths

    @staticmethod
    def _python_package_cutlass_paths():
        try:
            cutlass_library = importlib.import_module("cutlass_library")
        except ImportError:
            return []

        candidates = []
        source_path = getattr(cutlass_library, "source_path", None)
        if source_path is not None:
            candidates.append(Path(source_path))

        package_file = getattr(cutlass_library, "__file__", None)
        if package_file is not None:
            package_dir = Path(package_file).resolve().parent
            candidates.extend([package_dir / "source", package_dir.parent, package_dir])
        return candidates

    def _candidate_cutlass_paths(self):
        if self.cutlass_path == self.CUTLASS_PYTHON_BINDINGS:
            candidates = self._python_package_cutlass_paths()
            if candidates:
                return candidates
            self.warning("Please pip install nvidia-cutlass")
            raise ImportError("Unable to locate CUTLASS from the nvidia-cutlass Python package")

        if self.cutlass_path:
            return [Path(self.cutlass_path)]

        repo_root = self._repo_root()
        python_prefixes = self._dedupe_paths([Path(sys.prefix), Path(sys.exec_prefix), Path(sys.base_prefix)])
        prefix_paths = self._env_paths("CUTLASS_ROOT", "CUTLASS_HOME", "CONDA_PREFIX", "VIRTUAL_ENV",
                                       "CMAKE_PREFIX_PATH", "CUDA_HOME", "CUDA_PATH")
        include_paths = self._env_paths("CPATH", "CPLUS_INCLUDE_PATH", "C_INCLUDE_PATH")

        return self._dedupe_paths([
            *self._python_package_cutlass_paths(),
            *prefix_paths,
            *python_prefixes,
            *include_paths,
            Path.cwd() / "cutlass",
            repo_root / "cutlass",
            repo_root.parent / "cutlass",
            Path("/usr/local/cutlass"),
            Path("/opt/cutlass"),
            Path("/usr/local"),
            Path("/usr"),
        ])

    @staticmethod
    def _cutlass_include_dirs(cutlass_path):
        cutlass_path = cutlass_path.expanduser().resolve()
        if not cutlass_path.is_dir():
            return []

        if (cutlass_path / "include" / "cutlass" / "cutlass.h").is_file():
            include_root = cutlass_path / "include"
            util_include = cutlass_path / "tools" / "util" / "include"
        elif (cutlass_path / "cutlass" / "cutlass.h").is_file():
            include_root = cutlass_path
            util_include = cutlass_path.parent / "tools" / "util" / "include"
        else:
            return []

        include_dirs = [include_root]
        if util_include.is_dir():
            include_dirs.append(util_include)
        return [str(include_dir) for include_dir in include_dirs]

    def include_paths(self):
        # Assume the user knows best and CUTLASS location is already setup externally
        if self.cutlass_path == self.CUTLASS_IGNORE:
            return []

        for cutlass_path in self._candidate_cutlass_paths():
            include_dirs = self._cutlass_include_dirs(cutlass_path)
            if include_dirs:
                self._resolved_cutlass_path = cutlass_path.expanduser().resolve()
                return include_dirs

        if self.cutlass_path:
            raise RuntimeError(f"CUTLASS_PATH {self.cutlass_path} does not contain CUTLASS headers")

        raise RuntimeError("Unable to locate CUTLASS. Install nvidia-cutlass, clone CUTLASS next to DeepSpeed, "
                           "or set CUTLASS_PATH to the CUTLASS checkout.")
