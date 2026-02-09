# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import os
import time
import importlib

try:
    # is op_builder from deepspeed or a 3p version? this should only succeed if it's deepspeed
    # if successful this also means we're doing a local install and not JIT compile path
    from op_builder import __deepspeed__  # noqa: F401 # type: ignore
    from op_builder.builder import OpBuilder
except ImportError:
    from deepspeed.ops.op_builder.builder import OpBuilder


class SYCLOpBuilder(OpBuilder):

    def builder(self):
        from torch.utils.cpp_extension import SyclExtension
        include_dirs = [os.path.abspath(x) for x in self.strip_empty_entries(self.include_paths())]
        print("sycl sources = {}".format(self.sources()))
        sycl_ext = SyclExtension(name=self.absolute_name(),
                                 sources=self.strip_empty_entries(self.sources()),
                                 include_dirs=include_dirs,
                                 extra_compile_args={
                                     'cxx': self.strip_empty_entries(self.cxx_args()),
                                 },
                                 extra_link_args=self.strip_empty_entries(self.fixed_aotflags()))
        return sycl_ext

    def version_dependent_macros(self):
        try:
            from op_builder.builder import TORCH_MAJOR, TORCH_MINOR
        except ImportError:
            from deepspeed.ops.op_builder.builder import TORCH_MAJOR, TORCH_MINOR
        # Fix from apex that might be relevant for us as well, related to https://github.com/NVIDIA/apex/issues/456
        version_ge_1_1 = []
        if (TORCH_MAJOR > 1) or (TORCH_MAJOR == 1 and TORCH_MINOR > 0):
            version_ge_1_1 = ['-DVERSION_GE_1_1']
        version_ge_1_3 = []
        if (TORCH_MAJOR > 1) or (TORCH_MAJOR == 1 and TORCH_MINOR > 2):
            version_ge_1_3 = ['-DVERSION_GE_1_3']
        version_ge_1_5 = []
        if (TORCH_MAJOR > 1) or (TORCH_MAJOR == 1 and TORCH_MINOR > 4):
            version_ge_1_5 = ['-DVERSION_GE_1_5']
        return version_ge_1_1 + version_ge_1_3 + version_ge_1_5

    def _sycl_env_paths(self):
        """Find the SYCL include and lib directories from the Python environment.

        When using stock PyTorch XPU wheels, libsycl.so and SYCL headers are
        installed into the Python environment (e.g. conda env).  The system
        ``icpx`` compiler ships its own (potentially newer) SYCL headers and
        runtime.  To avoid ABI mismatches we must compile and link against the
        *same* SYCL version that PyTorch was built with.

        Returns (include_dir, lib_dir) – either or both may be ``None`` when
        the paths do not exist.
        """
        import sys
        prefix = sys.prefix  # e.g. /home/user/miniforge3/envs/myenv
        inc = os.path.join(prefix, 'include')
        lib = os.path.join(prefix, 'lib')
        sycl_inc = inc if os.path.isdir(os.path.join(inc, 'sycl')) else None
        sycl_lib = lib if os.path.isfile(os.path.join(lib, 'libsycl.so')) else None
        return sycl_inc, sycl_lib

    def cxx_args(self):
        cxx_flags = [
            '-fsycl', '-fsycl-targets=spir64_gen', '-g', '-gdwarf-4', '-O3', '-std=c++17', '-fPIC', '-DMKL_ILP64',
            '-fno-strict-aliasing'
        ]
        # Use SYCL headers from the Python environment so that compiled code
        # references symbols present in the *environment's* libsycl.so rather
        # than the (possibly newer) system oneAPI installation.
        sycl_inc, _ = self._sycl_env_paths()
        if sycl_inc:
            cxx_flags = [f'-isystem', sycl_inc] + cxx_flags
        if os.environ.get('USE_MKL_GEMM'):
            cxx_flags.append('-DUSE_MKL_GEMM')
        return cxx_flags

    def extra_ldflags(self):
        import torch
        torch_lib_dir = os.path.join(os.path.dirname(torch.__file__), 'lib')
        flags = [
            '-fPIC', '-fsycl', '-fsycl-targets=spir64_gen', '-fsycl-max-parallel-link-jobs=8',
            '-Xs "-options -cl-poison-unsupported-fp64-kernels,cl-intel-enable-auto-large-GRF-mode"',
            '-Xs "-device pvc"', '-Wl,-export-dynamic',
            f'-L{torch_lib_dir}', f'-Wl,-rpath,{torch_lib_dir}',
        ]
        # Link against the Python environment's libsycl.so to match the
        # headers we compiled against (see cxx_args).
        _, sycl_lib = self._sycl_env_paths()
        if sycl_lib:
            flags = [f'-L{sycl_lib}', f'-Wl,-rpath,{sycl_lib}'] + flags
        return flags

    def fixed_aotflags(self):
        return [
            '-fsycl', '-fsycl-targets=spir64_gen', '-fsycl-max-parallel-link-jobs=8', '-Xs',
            "-options -cl-poison-unsupported-fp64-kernels,cl-intel-enable-auto-large-GRF-mode", '-Xs', "-device pvc"
        ]

    def load(self, verbose=True):
        from deepspeed.git_version_info import installed_ops, torch_info, accelerator_name  # noqa: F401
        from deepspeed.accelerator import get_accelerator
        if installed_ops.get(self.name, False) and accelerator_name == get_accelerator()._name:
            return importlib.import_module(self.absolute_name())
        else:
            return self.jit_load(verbose)

    def jit_load(self, verbose=True):
        if not self.is_compatible(verbose):
            raise RuntimeError(
                f"Unable to JIT load the {self.name} op due to it not being compatible due to hardware/software issue. {self.error_log}"
            )
        from torch.utils.cpp_extension import verify_ninja_availability
        try:
            verify_ninja_availability()
        except RuntimeError as e:
            raise RuntimeError(f"Unable to JIT load the {self.name} op due to ninja not being installed.") from e

        self.jit_mode = True
        from torch.utils.cpp_extension import load

        start_build = time.time()
        # Recognize relative paths as absolute paths for jit load

        sources = [self.deepspeed_src_path(path) for path in self.sources()]
        extra_include_paths = [self.deepspeed_src_path(path) for path in self.include_paths()]

        # Set CXX to icpx (Intel oneAPI DPC++ compiler) so that .cpp/.dp.cpp
        # files containing SYCL code are compiled with the SYCL-aware compiler.
        # Stock PyTorch's cpp_extension only routes .sycl files to icpx by default.
        saved_env = {}
        for var in ('CXX', 'LIBRARY_PATH', 'CPATH'):
            saved_env[var] = os.environ.get(var)
        os.environ['CXX'] = 'icpx'

        # Point icpx at the Python environment's SYCL headers and libraries so
        # the compiled extension uses the same SYCL ABI as PyTorch.
        sycl_inc, sycl_lib = self._sycl_env_paths()
        if sycl_lib:
            lib_path = os.environ.get('LIBRARY_PATH', '')
            os.environ['LIBRARY_PATH'] = f'{sycl_lib}:{lib_path}' if lib_path else sycl_lib
        if sycl_inc:
            cpath = os.environ.get('CPATH', '')
            os.environ['CPATH'] = f'{sycl_inc}:{cpath}' if cpath else sycl_inc

        try:
            op_module = load(
                name=self.name,
                sources=self.strip_empty_entries(sources),
                extra_include_paths=self.strip_empty_entries(extra_include_paths),
                extra_cflags=self.strip_empty_entries(self.cxx_args()),
                extra_ldflags=self.strip_empty_entries(self.extra_ldflags()),
                verbose=verbose)
        finally:
            # Restore original environment
            for var, val in saved_env.items():
                if val is None:
                    os.environ.pop(var, None)
                else:
                    os.environ[var] = val

        build_duration = time.time() - start_build
        if verbose:
            print(f"Time to load {self.name} op: {build_duration} seconds")
        return op_module
