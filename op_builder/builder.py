# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import os
import re
import struct
import sys
import time
import importlib
import functools
import platform
from pathlib import Path
import subprocess
import shlex
import shutil
import tempfile
from abc import ABC, abstractmethod
from typing import List, NamedTuple

WARNING = "\033[93m [WARNING] \033[0m"

DEFAULT_COMPUTE_CAPABILITIES = "6.0;6.1;7.0"

try:
    import torch
except ImportError:
    print(f"{WARNING} unable to import torch, please install it if you want to pre-compile any deepspeed ops.")
else:
    TORCH_MAJOR = int(torch.__version__.split('.')[0])
    TORCH_MINOR = int(torch.__version__.split('.')[1])


class MissingCUDAException(Exception):
    pass


class CUDAMismatchException(Exception):
    pass


def installed_cuda_version():
    import torch.utils.cpp_extension
    cuda_home = torch.utils.cpp_extension.CUDA_HOME
    if cuda_home is None:
        raise MissingCUDAException("CUDA_HOME does not exist, unable to compile CUDA op(s)")
    # Ensure there is not a cuda version mismatch between torch and nvcc compiler
    return _nvcc_version(cuda_home + "/bin/nvcc")


# Every CUDA op asks for the toolkit version, so nvcc runs once per nvcc path and process. The cache holds just
# those (path, version) pairs, never a builder.
@functools.lru_cache(maxsize=None)
def _nvcc_version(nvcc):
    try:
        output = subprocess.check_output([nvcc, "-V"], universal_newlines=True)
    except (OSError, subprocess.SubprocessError) as err:
        # CUDA_HOME can point at a runtime only install, which has no nvcc to compile with. Report that
        # the same way as a missing CUDA_HOME so the callers that already fall back can handle it.
        raise MissingCUDAException(f"Unable to run {nvcc}, unable to compile CUDA op(s): {err}") from err
    output_split = output.split()
    release_idx = output_split.index("release")
    release = output_split[release_idx + 1].replace(',', '').split(".")
    # Ignore patch versions, only look at major + minor
    cuda_major, cuda_minor = release[:2]
    return int(cuda_major), int(cuda_minor)


# py-cpuinfo's names for the platform.machine() values that DeepSpeed selects CPU flags for.
_CPU_ARCH_NAMES = {
    'x86_64': 'X86_64',
    'amd64': 'X86_64',
    'aarch64': 'ARM_8',
    'arm64': 'ARM_8',
    'ppc64': 'PPC_64',
    'ppc64le': 'PPC_64',
    'riscv64': 'RISCV_64',
}


@functools.lru_cache(maxsize=None)
def _cpu_info():
    '''
    Returns the host CPU architecture, named the way py-cpuinfo names it (X86_64, ARM_8, PPC_64, RISCV_64, ...),
    and its feature flags as a frozenset. The CPU is queried once per process, through py-cpuinfo or, when that is
    missing or fails, lscpu. Returns (None, frozenset()) when neither works.
    '''
    try:
        from cpuinfo import get_cpu_info
    except ImportError:
        return _lscpu_cpu_info()
    try:
        cpu_info = get_cpu_info()
    except Exception as e:
        print(f"{WARNING} py-cpuinfo failed to detect the CPU (exception type: {type(e)}, {e}), "
              "falling back to lscpu to get this information.")
        return _lscpu_cpu_info()
    return cpu_info.get('arch'), frozenset(cpu_info.get('flags', []))


def _lscpu_cpu_info():
    try:
        # LC_ALL=C keeps the field names in English.
        output = subprocess.check_output(['lscpu'], universal_newlines=True, env={**os.environ, 'LC_ALL': 'C'})
    except (OSError, subprocess.SubprocessError) as e:
        print(f"{WARNING} Unable to detect the CPU with py-cpuinfo or lscpu ({type(e).__name__}: {e}), "
              "will fall back to use -march=native and non-vectorized execution.")
        return None, frozenset()
    flags = ''
    for line in output.splitlines():
        if line.startswith('Flags:'):
            flags = line[len('Flags:'):]
    return _CPU_ARCH_NAMES.get(platform.machine().lower()), frozenset(flags.lower().split())


def probe_is_compatible(builder, verbose=False):
    """Report whether an op can be built, treating a probe that itself fails as "not compatible".

    Compatibility is probed for every op when deepspeed is imported, whether or not the caller will ever
    build that op. A CUDA op builder on a machine that has a GPU but no CUDA toolkit raises
    MissingCUDAException from its probe, which would otherwise make importing deepspeed fail outright
    over an op that was never asked for. See #7452.
    """
    try:
        return builder.is_compatible(verbose)
    except Exception as err:
        print(f"{WARNING} {builder.name} compatibility check failed ({err}), treating the op as not compatible")
        return False


def get_default_compute_capabilities():
    compute_caps = DEFAULT_COMPUTE_CAPABILITIES
    # Update compute capability according to: https://en.wikipedia.org/wiki/CUDA#GPUs_supported
    import torch.utils.cpp_extension
    if torch.utils.cpp_extension.CUDA_HOME is not None:
        if installed_cuda_version()[0] == 11:
            if installed_cuda_version()[1] >= 0:
                compute_caps += ";8.0"
            if installed_cuda_version()[1] >= 1:
                compute_caps += ";8.6"
            if installed_cuda_version()[1] >= 8:
                compute_caps += ";9.0"
        elif installed_cuda_version()[0] == 12:
            compute_caps += ";8.0;8.6;9.0"
            if installed_cuda_version()[1] >= 8:
                compute_caps += ";10.0;12.0"
        elif installed_cuda_version()[0] >= 13:
            # CUDA 13 cannot build for compute capabilities below 7.5, which rules out the whole base list
            compute_caps = "7.5;8.0;8.6;9.0;10.0;12.0"
    return compute_caps


class CudaArch(NamedTuple):
    """One TORCH_CUDA_ARCH_LIST entry, such as 8.0, 9.0a or 10.0+PTX."""
    major: int
    minor: int
    # "a" marks an architecture-specific target, whose code only runs on that exact GPU.
    suffix: str
    ptx: bool

    def __str__(self):
        return f"{self.major}.{self.minor}{self.suffix}{'+PTX' if self.ptx else ''}"


def _parse_cuda_arch_list(arch_list):
    """
    Parse a TORCH_CUDA_ARCH_LIST value, such as "8.0;9.0a 10.0+PTX", into CudaArch entries.

    Entries are separated by ; or whitespace, and empty entries are ignored. Like PyTorch's extension
    builder, the only suffix accepted is "a". Architecture names such as "Ampere" are rejected because
    they are ambiguous.
    """
    ccs = []
    for entry in arch_list.replace(';', ' ').split():
        match = re.fullmatch(r"(\d+)\.(\d+)(a?)(\+PTX)?", entry)
        if match is None:
            raise ValueError(f"Unsupported TORCH_CUDA_ARCH_LIST entry '{entry}'. Use compute capabilities such as "
                             "8.0, 9.0a or 10.0+PTX.")
        major, minor, suffix, ptx = match.groups()
        ccs.append(CudaArch(int(major), int(minor), suffix, ptx is not None))
    return ccs


def assert_no_cuda_mismatch():
    cuda_major, cuda_minor = installed_cuda_version()
    sys_cuda_version = f'{cuda_major}.{cuda_minor}'
    torch_cuda_version = ".".join(torch.version.cuda.split('.')[:2])
    if sys_cuda_version != torch_cuda_version:
        torch_cuda_major = int(torch_cuda_version.split('.')[0])
        # Like PyTorch's own extension build, fail only on a major version mismatch: CUDA keeps the
        # releases of one major version compatible, so a minor mismatch only gets a warning.
        if cuda_major == torch_cuda_major:
            print(f"{WARNING} DeepSpeed Op Builder: Installed CUDA version {sys_cuda_version} has a minor version "
                  f"mismatch with the version torch was compiled with {torch.version.cuda}. Accepting this "
                  "combination, but it may result in unexpected behavior.")
            return True
        elif os.getenv("DS_SKIP_CUDA_CHECK", "0") == "1":
            print(
                f"{WARNING} DeepSpeed Op Builder: Installed CUDA version {sys_cuda_version} does not match the "
                f"version torch was compiled with {torch.version.cuda}."
                "Detected `DS_SKIP_CUDA_CHECK=1`: Allowing this combination of CUDA, but it may result in unexpected behavior."
            )
            return True
        raise CUDAMismatchException(
            f">- DeepSpeed Op Builder: Installed CUDA version {sys_cuda_version} does not match the "
            f"version torch was compiled with {torch.version.cuda}, unable to compile "
            "cuda/cpp extensions without a matching cuda version.")
    return True


class OpBuilder(ABC):
    _is_rocm_pytorch = None
    _loaded_ops = {}

    def __init__(self, name):
        self.name = name
        self.error_log = None

    @abstractmethod
    def absolute_name(self):
        '''
        Returns absolute build path for cases where the op is pre-installed, e.g., deepspeed.ops.adam.cpu_adam
        will be installed as something like: deepspeed/ops/adam/cpu_adam.so
        '''
        pass

    @abstractmethod
    def sources(self):
        '''
        Returns list of source files for your op, relative to root of deepspeed package (i.e., DeepSpeed/deepspeed)
        '''
        pass

    def hipify_extension(self):
        pass

    @staticmethod
    def validate_torch_version(torch_info):
        install_torch_version = torch_info['version']
        current_torch_version = ".".join(torch.__version__.split('.')[:2])
        if install_torch_version != current_torch_version:
            raise RuntimeError("PyTorch version mismatch! DeepSpeed ops were compiled and installed "
                               "with a different version than what is being used at runtime. "
                               f"Please re-install DeepSpeed or switch torch versions. "
                               f"Install torch version={install_torch_version}, "
                               f"Runtime torch version={current_torch_version}")

    def validate_torch_op_version(self, torch_info):
        '''
        Check that a pre-compiled op was built against the device toolkit of the runtime torch. Builders
        whose ops are compiled against a device toolkit override this.
        '''
        pass

    @staticmethod
    def is_rocm_pytorch():
        if OpBuilder._is_rocm_pytorch is not None:
            return OpBuilder._is_rocm_pytorch

        _is_rocm_pytorch = False
        try:
            import torch
        except ImportError:
            pass
        else:
            _is_rocm_pytorch = hasattr(torch.version, 'hip') and torch.version.hip is not None
            if _is_rocm_pytorch:
                from torch.utils.cpp_extension import ROCM_HOME
                _is_rocm_pytorch = ROCM_HOME is not None
        OpBuilder._is_rocm_pytorch = _is_rocm_pytorch
        return OpBuilder._is_rocm_pytorch

    def include_paths(self):
        '''
        Returns list of include paths, relative to root of deepspeed package (i.e., DeepSpeed/deepspeed)
        '''
        return []

    def cxx_args(self):
        '''
        Returns optional list of compiler flags to forward to the build
        '''
        return []

    def is_compatible(self, verbose=False):
        '''
        Check if all non-python dependencies are satisfied to build this op
        '''
        return True

    def extra_ldflags(self):
        return []

    def libraries_args(self):
        '''
        Returns list of libraries to link the op against
        '''
        return []

    def extra_compile_args(self):
        '''
        Returns the flags for each compiler. builder() and jit_load() both build with them, so a pre-built op
        is compiled the same way as one compiled at load time.
        '''
        return {'cxx': self.strip_empty_entries(self.cxx_args())}

    def has_function(self, funcname, libraries, library_dirs=None, verbose=False):
        '''
        Test for existence of a function within a tuple of libraries.

        This is used as a smoke test to check whether a certain library is available.
        As a test, this creates a simple C program that calls the specified function,
        and then distutils is used to compile that program and link it with the specified libraries.
        Returns True if both the compile and link are successful, False otherwise.
        '''
        # Imported here because distutils pulls in setuptools, which slows down import deepspeed.
        import distutils.ccompiler
        import distutils.log
        import distutils.sysconfig

        tempdir = None  # we create a temporary directory to hold various files
        filestderr = None  # handle to open file to which we redirect stderr
        oldstderr = None  # file descriptor for stderr
        try:
            # Echo compile and link commands that are used.
            if verbose:
                distutils.log.set_verbosity(1)

            # Create a compiler object.
            compiler = distutils.ccompiler.new_compiler(verbose=verbose)

            # Configure compiler and linker to build according to Python install.
            distutils.sysconfig.customize_compiler(compiler)

            # Create a temporary directory to hold test files.
            tempdir = tempfile.mkdtemp()

            # Define a simple C program that calls the function in question
            prog = "void %s(void); int main(int argc, char** argv) { %s(); return 0; }" % (funcname, funcname)

            # Write the test program to a file.
            filename = os.path.join(tempdir, 'test.c')
            with open(filename, 'w') as f:
                f.write(prog)

            # Redirect stderr file descriptor to a file to silence compile/link warnings.
            if not verbose:
                filestderr = open(os.path.join(tempdir, 'stderr.txt'), 'w')
                oldstderr = os.dup(sys.stderr.fileno())
                os.dup2(filestderr.fileno(), sys.stderr.fileno())

            # Workaround for behavior in distutils.ccompiler.CCompiler.object_filenames()
            # Otherwise, a local directory will be used instead of tempdir
            drive, driveless_filename = os.path.splitdrive(filename)
            root_dir = driveless_filename[0] if os.path.isabs(driveless_filename) else ''
            output_dir = os.path.join(drive, root_dir)

            # Attempt to compile the C program into an object file.
            cflags = shlex.split(os.environ.get('CFLAGS', ""))
            objs = compiler.compile([filename], output_dir=output_dir, extra_preargs=self.strip_empty_entries(cflags))

            # Attempt to link the object file into an executable.
            # Be sure to tack on any libraries that have been specified.
            ldflags = shlex.split(os.environ.get('LDFLAGS', ""))
            compiler.link_executable(objs,
                                     os.path.join(tempdir, 'a.out'),
                                     extra_preargs=self.strip_empty_entries(ldflags),
                                     libraries=libraries,
                                     library_dirs=library_dirs)

            # Compile and link succeeded
            return True

        except Exception:
            return False

        finally:
            # Restore stderr file descriptor and close the stderr redirect file.
            if oldstderr is not None:
                os.dup2(oldstderr, sys.stderr.fileno())
                os.close(oldstderr)
            if filestderr is not None:
                filestderr.close()

            # Delete the temporary directory holding the test program and stderr files.
            if tempdir is not None:
                shutil.rmtree(tempdir)

    def strip_empty_entries(self, args):
        '''
        Drop any empty strings from the list of compile and link flags
        '''
        return [x for x in args if len(x) > 0]

    def cpu_arch(self):
        arch, _ = _cpu_info()
        if (arch or '').startswith('PPC_'):
            # gcc does not provide -march on PowerPC, use -mcpu instead
            return '-mcpu=native'
        elif arch == 'RISCV_64':
            # The vector extension must be in -march for the RVV kernels to compile.
            return '-march=rv64gcv' if self._riscv_has_vector() else '-march=rv64gc'
        return '-march=native'

    def _riscv_hwcap(self):
        # AT_HWCAP from the auxiliary vector, the same source getauxval(3) serves.
        # The kernel builds ELF_HWCAP from the extensions its own ISA parsing
        # accepted, so unlike a raw ISA string the V bit does not mirror draft
        # tokens. None means the kernel provided no AT_HWCAP at all.
        try:
            with open('/proc/self/auxv', 'rb') as f:
                auxv = f.read()
        except OSError:
            return None
        at_hwcap = 16  # <linux/auxvec.h>
        for offset in range(0, len(auxv) - 15, 16):
            a_type, value = struct.unpack('=QQ', auxv[offset:offset + 16])
            if a_type == at_hwcap:
                return value
        return None

    def _riscv_has_vector(self):
        # A bare "v" in the ISA line is not evidence of ratified V 1.0: T-Head C906
        # (Allwinner D1) devicetrees report unversioned rv64imafdcv while the core
        # implements incompatible draft V 0.7.1, and -march=rv64gcv would emit
        # encodings such a CPU cannot execute. An explicitly versioned token is
        # authoritative ("v1p0" qualifies, "v0p7" never does); anything else defers
        # to the kernel's HWCAP V bit, the availability interface the kernel's own
        # vector documentation defines. Only the base segment before the first
        # underscore carries single-letter V (zvl*b, zvfh, ... can never match), and
        # there each letter is its own token, optionally versioned ("i2p0" style).
        try:
            with open('/proc/cpuinfo') as f:
                for line in f:
                    if line.startswith('isa'):
                        base = line.split(':', 1)[1].split('_', 1)[0].strip()[4:]
                        for token in re.findall(r'[a-z](?:\d+p\d+)?', base):
                            if token[0] == 'v' and token[1:]:
                                return int(token[1:].partition('p')[0]) >= 1
                        break
        except OSError:
            pass
        hwcap = self._riscv_hwcap()
        return hwcap is not None and hwcap & (1 << (ord('V') - ord('A'))) != 0  # COMPAT_HWCAP_ISA_V

    def simd_width(self):
        arch, flags = _cpu_info()
        if arch == 'X86_64':
            if 'avx512f' in flags:
                return '-D__AVX512__'
            elif 'avx2' in flags:
                return '-D__AVX256__'
        elif (arch or '').startswith('ARM') and 'sve' in flags:
            return '-D__SVE__'
        elif arch == 'ARM_8':
            # NEON is baseline on AArch64; vdivq/vsqrtq used by simd.h are A64-only,
            # so do not advertise it for 32-bit ARM.
            return '-D__NEON__'
        elif arch == 'RISCV_64' and self._riscv_has_vector():
            return '-D__RVV__'
        return '-D__SCALAR__'

    def warning(self, msg):
        self.error_log = f"{msg}"
        print(f"{WARNING} {msg}")

    def deepspeed_src_path(self, code_path):
        if os.path.isabs(code_path):
            return code_path
        else:
            return os.path.join(Path(__file__).parent.parent.absolute(), code_path)

    def absolute_include_paths(self):
        return [
            os.path.abspath(self.deepspeed_src_path(path)) for path in self.strip_empty_entries(self.include_paths())
        ]

    def builder(self):
        from torch.utils.cpp_extension import CppExtension
        include_dirs = self.absolute_include_paths()
        return CppExtension(name=self.absolute_name(),
                            sources=self.strip_empty_entries(self.sources()),
                            include_dirs=include_dirs,
                            libraries=self.strip_empty_entries(self.libraries_args()),
                            extra_compile_args=self.extra_compile_args(),
                            extra_link_args=self.strip_empty_entries(self.extra_ldflags()))

    def load(self, verbose=False):
        if self.name in __class__._loaded_ops:
            return __class__._loaded_ops[self.name]

        from deepspeed.git_version_info import installed_ops, torch_info, accelerator_name
        from deepspeed.accelerator import get_accelerator
        if installed_ops.get(self.name, False) and accelerator_name == get_accelerator()._name:
            # Ensure the op we're about to load was compiled with the same
            # torch and device toolkit versions we are currently using at runtime.
            self.validate_torch_version(torch_info)
            self.validate_torch_op_version(torch_info)

            op_module = importlib.import_module(self.absolute_name())
            __class__._loaded_ops[self.name] = op_module
            return op_module
        else:
            return self.jit_load(verbose)

    def jit_load_kwargs(self):
        '''
        Returns the compile arguments that jit_load() passes to torch.utils.cpp_extension.load()
        '''
        return {'extra_cflags': self.extra_compile_args()['cxx']}

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

        from torch.utils.cpp_extension import load

        start_build = time.time()
        sources = [os.path.abspath(self.deepspeed_src_path(path)) for path in self.sources()]
        extra_include_paths = self.absolute_include_paths()

        compile_kwargs = self.jit_load_kwargs()
        extra_ldflags = self.strip_empty_entries(self.extra_ldflags())
        # load() has no libraries argument, so link them by name the way setuptools does for builder().
        for library in self.strip_empty_entries(self.libraries_args()):
            extra_ldflags.append(f"{library}.lib" if sys.platform == "win32" else f"-l{library}")
        op_module = load(name=self.name,
                         sources=self.strip_empty_entries(sources),
                         extra_include_paths=self.strip_empty_entries(extra_include_paths),
                         extra_ldflags=extra_ldflags,
                         verbose=verbose,
                         **compile_kwargs)

        build_duration = time.time() - start_build
        if verbose:
            print(f"Time to load {self.name} op: {build_duration} seconds")

        __class__._loaded_ops[self.name] = op_module

        return op_module


class CUDAOpBuilder(OpBuilder):
    _rocm_version = None
    # Oldest compute capability major the op supports, or None to skip these checks. For an op that sets it,
    # filter_ccs() drops older build targets, and is_compatible() requires torch and, when a GPU is visible, a CUDA
    # toolkit and a GPU that is not older.
    MIN_COMPUTE_CAPABILITY = None

    def __init__(self, name):
        super().__init__(name)
        # Ops are JIT-built for the GPUs of the machine that loads them, unless builder() pre-builds them.
        self.jit_mode = True
        self.enable_bf16 = False

    def validate_torch_op_version(self, torch_info):
        if not torch.cuda.is_available():
            return
        if not self.is_rocm_pytorch():
            current_cuda_version = ".".join(torch.version.cuda.split('.')[:2])
            install_cuda_version = torch_info['cuda_version']
            if install_cuda_version != current_cuda_version:
                raise RuntimeError("CUDA version mismatch! DeepSpeed ops were compiled and installed "
                                   "with a different version than what is being used at runtime. "
                                   f"Please re-install DeepSpeed or switch torch versions. "
                                   f"Install CUDA version={install_cuda_version}, "
                                   f"Runtime CUDA version={current_cuda_version}")
        else:
            current_hip_version = ".".join(torch.version.hip.split('.')[:2])
            install_hip_version = torch_info['hip_version']
            if install_hip_version != current_hip_version:
                raise RuntimeError("HIP version mismatch! DeepSpeed ops were compiled and installed "
                                   "with a different version than what is being used at runtime. "
                                   f"Please re-install DeepSpeed or switch torch versions. "
                                   f"Install HIP version={install_hip_version}, "
                                   f"Runtime HIP version={current_hip_version}")

    @staticmethod
    def installed_rocm_version():
        if CUDAOpBuilder._rocm_version:
            return CUDAOpBuilder._rocm_version

        ROCM_MAJOR = '0'
        ROCM_MINOR = '0'
        ROCM_VERSION_DEV_RAW = ""
        if OpBuilder.is_rocm_pytorch():
            from torch.utils.cpp_extension import ROCM_HOME
            rocm_ver_file = Path(ROCM_HOME).joinpath(".info/version")
            if rocm_ver_file.is_file():
                with open(rocm_ver_file, 'r') as file:
                    ROCM_VERSION_DEV_RAW = file.read()
            elif "rocm" in torch.__version__:
                ROCM_VERSION_DEV_RAW = torch.__version__.split("rocm")[1]
            if ROCM_VERSION_DEV_RAW != "":
                ROCM_MAJOR = ROCM_VERSION_DEV_RAW.split('.')[0]
                ROCM_MINOR = ROCM_VERSION_DEV_RAW.split('.')[1]
            else:
                # Look in /usr/include/rocm-version.h
                rocm_ver_file = Path("/usr/include/rocm_version.h")
                if rocm_ver_file.is_file():
                    with open(rocm_ver_file, 'r') as file:
                        for ln in file.readlines():
                            if "#define ROCM_VERSION_MAJOR" in ln:
                                ROCM_MAJOR = re.findall(r'\S+', ln)[2]
                            elif "#define ROCM_VERSION_MINOR" in ln:
                                ROCM_MINOR = re.findall(r'\S+', ln)[2]
            if ROCM_MAJOR == '0':
                assert False, "Could not detect ROCm version"

        CUDAOpBuilder._rocm_version = (int(ROCM_MAJOR), int(ROCM_MINOR))
        return CUDAOpBuilder._rocm_version

    @staticmethod
    @functools.lru_cache(maxsize=None)
    def _rocminfo_output():
        # The GPU arch and the wavefront size both come from rocminfo, so it runs once per process. An empty
        # output, when rocminfo cannot run, is cached as well.
        rocm_info = Path("/opt/rocm/bin/rocminfo")
        if (not rocm_info.is_file()):
            rocm_info = Path("rocminfo")
        try:
            result = subprocess.check_output([str(rocm_info)], stderr=subprocess.DEVNULL)
            return result.decode('utf-8')
        except (subprocess.CalledProcessError, FileNotFoundError, OSError):
            return ""

    @staticmethod
    def get_rocm_gpu_arch():
        match = re.search(r'gfx\S+', CUDAOpBuilder._rocminfo_output())
        return match.group(0).strip() if match else ""

    @staticmethod
    def get_rocm_wavefront_size():
        match = re.search(r'Wavefront Size:\s+(\d+)', CUDAOpBuilder._rocminfo_output())
        return match.group(1) if match else "32"

    def cuda_capability_major(self):
        """Compute-capability major of CUDA device 0, or ``None`` when it cannot
        be read without side effects.

        ``torch.cuda.get_device_properties`` calls ``torch.cuda._lazy_init()``,
        which creates a CUDA context. Doing that merely to check op compatibility
        at ``import deepspeed`` time would poison ``fork()``-based multiprocessing,
        because a forked child cannot reuse the parent's context (issue #7918).
        We therefore probe only when a context already exists and we are not
        inside such a forked child; otherwise the caller skips the
        compute-capability check and defers it to build/load time.
        """
        if not torch.cuda.is_initialized():
            return None
        if hasattr(torch.cuda, '_is_in_bad_fork') and torch.cuda._is_in_bad_fork():
            return None
        return torch.cuda.get_device_properties(0).major

    def compute_capability_args(self):
        """
        Returns nvcc compute capability compile flags.

        1. Under ``jit_mode``, the precedence is:
           a. preserved ``TORCH_CUDA_ARCH_LIST`` captured by ``jit_load()``
           b. live ``TORCH_CUDA_ARCH_LIST`` from the environment
           c. runtime device probing when the process is not in a bad-fork context
           d. an error when no explicit arch list exists in a bad-fork context

           JIT mode then sets ``TORCH_CUDA_ARCH_LIST`` so PyTorch can generate
           the ``-gencode`` flags itself.
        2. Otherwise ``TORCH_CUDA_ARCH_LIST`` is used.
        3. If it is not set, default compute capabilities will be used.
        4. In both modes, when no entry carries ``+PTX``, the highest entry
           without a suffix gets it.

        Format:

        - ``TORCH_CUDA_ARCH_LIST`` may use ; or whitespace separators, ``+PTX``,
          and the ``a`` suffix. Examples:

        TORCH_CUDA_ARCH_LIST="6.1;7.5;8.6;9.0;10.0" pip install ...
        TORCH_CUDA_ARCH_LIST="6.0 6.1 7.0 7.5 8.0 8.6 9.0 10.0+PTX" pip install ...

        """
        if self.jit_mode:
            arch_list = getattr(self, '_jit_arch_list', None) or os.environ.get('TORCH_CUDA_ARCH_LIST', '')
            ccs = _parse_cuda_arch_list(arch_list)
            if not ccs:
                if hasattr(torch.cuda, '_is_in_bad_fork') and torch.cuda._is_in_bad_fork():
                    raise RuntimeError(
                        f"DeepSpeed JIT builder for '{self.name}' cannot probe CUDA device capabilities "
                        "in a forked subprocess where CUDA has already been initialized. Set "
                        "TORCH_CUDA_ARCH_LIST to specify target architectures explicitly.")
                for i in range(torch.cuda.device_count()):
                    CC_MAJOR, CC_MINOR = torch.cuda.get_device_capability(i)
                    cc = CudaArch(CC_MAJOR, CC_MINOR, '', False)
                    if cc not in ccs:
                        ccs.append(cc)
                if len(ccs) == 0:
                    raise RuntimeError(f"DeepSpeed JIT builder for '{self.name}' found no CUDA devices. Set "
                                       "TORCH_CUDA_ARCH_LIST or make GPUs visible.")
        else:
            # Cross-compile mode, compile for various architectures
            # env override takes priority
            ccs = _parse_cuda_arch_list(os.environ.get('TORCH_CUDA_ARCH_LIST', ''))
            if not ccs:
                ccs = _parse_cuda_arch_list(get_default_compute_capabilities())

        ccs = self.filter_ccs(ccs)
        if len(ccs) == 0:
            raise RuntimeError(
                f"Unable to load {self.name} op due to no compute capabilities remaining after filtering")

        # Canonicalize by (major, minor, suffix) so the emitted -gencode
        # sequence matches PyTorch's own dedupe (see #7871). For mixed inputs
        # such as "8.0;8.0+PTX" or "8.0+PTX;8.0", PyTorch collapses to one
        # sm_80 entry plus one compute_80 PTX entry. Track has_PTX per arch
        # as the OR across all variants of that arch so any +PTX appearance
        # carries through after dedupe.
        canonical = {}
        for cc in ccs:
            key = cc._replace(ptx=False)
            canonical[key] = canonical.get(key, False) or cc.ptx
        ccs = sorted(key._replace(ptx=has_ptx) for key, has_ptx in canonical.items())

        # PTX lets the op run on GPUs newer than every target. PTX for an "a" target only runs on that exact
        # GPU, so when no entry asks for PTX, add it to the highest target without a suffix.
        if not any(cc.ptx for cc in ccs):
            plain_ccs = [cc for cc in ccs if not cc.suffix]
            if plain_ccs:
                highest = plain_ccs[-1]
                ccs[ccs.index(highest)] = highest._replace(ptx=True)

        # Ampere (8.x) and newer GPUs have native bf16 arithmetic. For older targets, nvcc 12.2+ computes it in
        # float instead, while older toolkits fail to compile the BF16 kernels for them.
        self.enable_bf16 = all(cc.major >= 8 for cc in ccs)
        if not self.enable_bf16:
            from torch.utils.cpp_extension import CUDA_HOME
            self.enable_bf16 = CUDA_HOME is not None and installed_cuda_version() >= (12, 2)

        if self.jit_mode:
            # Before torch 2.9, load() adds -gencode flags from TORCH_CUDA_ARCH_LIST (or the visible GPUs) even
            # when extra_cuda_cflags already has some. So set the env var to the filtered arch list and let
            # PyTorch generate the flags, which keeps it from re-adding archs that filter_ccs() removed;
            # jit_load() restores the caller's value. Emit one token per arch using the X.Y[a] or X.Y[a]+PTX
            # form, matching PyTorch's canonical parsing where +PTX on an arch token already implies both the
            # sm and PTX emissions for that arch.
            os.environ["TORCH_CUDA_ARCH_LIST"] = ";".join(str(cc) for cc in ccs)
            return []

        # Non-JIT: return explicit flags per builder for extra_compile_args.
        # BuildExtension uses these instead of TORCH_CUDA_ARCH_LIST, so leave the env var alone: setup.py
        # builds every op in one process, and each op must start from the requested arch list.
        # Emit exactly one sm_X line per arch, followed by one compute_X PTX
        # line when any variant of that arch carried +PTX.
        args = []
        for cc in ccs:
            num = f"{cc.major}{cc.minor}{cc.suffix}"
            args.append(f'-gencode=arch=compute_{num},code=sm_{num}')
            if cc.ptx:
                args.append(f'-gencode=arch=compute_{num},code=compute_{num}')

        return args

    def filter_ccs(self, ccs: List[CudaArch]):
        """
        Prune any compute capabilities that are not compatible with the builder. Should log
        which CCs have been pruned.
        """
        if self.MIN_COMPUTE_CAPABILITY is None:
            return ccs
        ccs_retained = [cc for cc in ccs if cc.major >= self.MIN_COMPUTE_CAPABILITY]
        ccs_pruned = [str(cc) for cc in ccs if cc.major < self.MIN_COMPUTE_CAPABILITY]
        if len(ccs_pruned) > 0:
            self.warning(f"Filtered compute capabilities {ccs_pruned}")
        return ccs_retained

    def is_compatible(self, verbose=False):
        if self.MIN_COMPUTE_CAPABILITY is None:
            return super().is_compatible(verbose)
        # setup.py checks every op, even when torch is not installed.
        try:
            import torch
        except ImportError:
            self.error_log = f"Please install torch if trying to pre-compile {self.name}"
            if verbose:
                self.warning(self.error_log)
            return False
        check_gpu = not os.environ.get("DS_IGNORE_CUDA_DETECTION") and not self.is_rocm_pytorch()
        if check_gpu and torch.cuda.is_available():
            # Raises MissingCUDAException when there is no CUDA toolkit to build the op with.
            installed_cuda_version()
            cuda_capability = self.cuda_capability_major()
            if cuda_capability is not None and cuda_capability < self.MIN_COMPUTE_CAPABILITY:
                # Recorded even when not verbose: load() checks with verbose=False, and jit_load() then reports
                # error_log as the reason.
                self.error_log = (f"{self.name} requires a GPU with compute capability "
                                  f"{self.MIN_COMPUTE_CAPABILITY}.0 or newer")
                if verbose:
                    self.warning(self.error_log)
                return False
        return super().is_compatible(verbose)

    def builder(self):
        # Pre-built ops run on other machines, so they target the requested arch list rather than
        # the GPUs of this one.
        self.jit_mode = False
        compile_args = self.extra_compile_args()
        if self.is_rocm_pytorch():
            # BuildExtension compiles hip files with the nvcc args only, while load() also passes them the cxx args.
            compile_args['nvcc'].append('-DROCM_WAVEFRONT_SIZE=%s' % self.get_rocm_wavefront_size())
            if self.get_rocm_gpu_arch():
                os.environ["PYTORCH_ROCM_ARCH"] = self.get_rocm_gpu_arch()

        from torch.utils.cpp_extension import CUDAExtension as ExtensionBuilder
        include_dirs = self.absolute_include_paths()
        cuda_ext = ExtensionBuilder(name=self.absolute_name(),
                                    sources=self.strip_empty_entries(self.sources()),
                                    include_dirs=include_dirs,
                                    libraries=self.strip_empty_entries(self.libraries_args()),
                                    extra_compile_args=compile_args,
                                    extra_link_args=self.strip_empty_entries(self.extra_ldflags()))

        if self.is_rocm_pytorch():
            # hip converts paths to absolute, this converts back to relative
            sources = cuda_ext.sources
            curr_file = Path(__file__).parent.parent  # ds root
            for i in range(len(sources)):
                src = Path(sources[i])
                if src.is_absolute():
                    sources[i] = str(src.relative_to(curr_file))
                else:
                    sources[i] = str(src)
            cuda_ext.sources = sources
        return cuda_ext

    def jit_load(self, verbose=True):
        # compute_capability_args() publishes the JIT arch list through TORCH_CUDA_ARCH_LIST for
        # PyTorch's build to read, so the caller's value is put back afterwards.
        torch_arch_list_present = "TORCH_CUDA_ARCH_LIST" in os.environ
        torch_arch_list = os.environ.get("TORCH_CUDA_ARCH_LIST")
        normalized_arch_list = torch_arch_list.strip() if torch_arch_list is not None else None
        self._jit_arch_list = normalized_arch_list or None
        saved_jit_mode = self.jit_mode
        self.jit_mode = True
        try:
            return super().jit_load(verbose)
        finally:
            if torch_arch_list_present:
                os.environ["TORCH_CUDA_ARCH_LIST"] = torch_arch_list
            else:
                os.environ.pop("TORCH_CUDA_ARCH_LIST", None)
            self._jit_arch_list = None
            self.jit_mode = saved_jit_mode

    def extra_compile_args(self):
        # Unlike BuildExtension, PyTorch's load() does not check the CUDA toolkit version, so check it here
        # for both build paths.
        if not self.is_rocm_pytorch():
            assert_no_cuda_mismatch()
        # nvcc_args() goes first because it decides enable_bf16 and the arch list.
        nvcc_args = self.strip_empty_entries(self.nvcc_args())
        compile_args = super().extra_compile_args()
        cxx_args = compile_args['cxx']

        if self.enable_bf16:
            cxx_args.append("-DBF16_AVAILABLE")
            nvcc_args.append("-DBF16_AVAILABLE")
            nvcc_args.append("-U__CUDA_NO_BFLOAT16_OPERATORS__")
            nvcc_args.append("-U__CUDA_NO_BFLOAT162_OPERATORS__")
            nvcc_args.append("-U__CUDA_NO_BFLOAT16_CONVERSIONS__")

        if self.is_rocm_pytorch():
            cxx_args.append("-D__HIP_PLATFORM_AMD__=1")
            cxx_args.append('-DROCM_WAVEFRONT_SIZE=%s' % self.get_rocm_wavefront_size())

        compile_args['nvcc'] = nvcc_args
        return compile_args

    def jit_load_kwargs(self):
        compile_args = self.extra_compile_args()
        if self.is_rocm_pytorch():
            # A JIT build targets the GPU of this machine.
            os.environ["PYTORCH_ROCM_ARCH"] = self.get_rocm_gpu_arch()
        return {'extra_cflags': compile_args['cxx'], 'extra_cuda_cflags': compile_args['nvcc'], 'with_cuda': True}

    def hipify_extension(self):
        if self.is_rocm_pytorch():
            from torch.utils.hipify import hipify_python
            hipify_python.hipify(
                project_directory=os.getcwd(),
                output_directory=os.getcwd(),
                header_include_dirs=self.include_paths(),
                includes=[os.path.join(os.getcwd(), '*')],
                extra_files=[os.path.abspath(s) for s in self.sources()],
                show_detailed=True,
                is_pytorch_extension=True,
                hipify_extra_files_only=True,
            )

    def cxx_args(self):
        # No -std flag: PyTorch adds the C++ standard its headers need, and a later -std would override it.
        if sys.platform == "win32":
            return ['-O2']
        else:
            return ['-O3', '-g', '-Wno-reorder']

    def nvcc_args(self):
        args = ['-O3']
        if self.is_rocm_pytorch():
            ROCM_MAJOR, ROCM_MINOR = self.installed_rocm_version()
            args += [
                '-U__HIP_NO_HALF_OPERATORS__', '-U__HIP_NO_HALF_CONVERSIONS__', '-U__HIP_NO_HALF2_OPERATORS__',
                '-DROCM_VERSION_MAJOR=%s' % ROCM_MAJOR,
                '-DROCM_VERSION_MINOR=%s' % ROCM_MINOR
            ]
            self.enable_bf16 = True
        else:
            try:
                nvcc_threads = int(os.getenv("DS_NVCC_THREADS", ""))
                if nvcc_threads <= 0:
                    raise ValueError("")
            except ValueError:
                nvcc_threads = min(os.cpu_count(), 8)

            args += [
                '-allow-unsupported-compiler' if sys.platform == "win32" else '', '--use_fast_math',
                '-U__CUDA_NO_HALF_OPERATORS__', '-U__CUDA_NO_HALF_CONVERSIONS__', '-U__CUDA_NO_HALF2_OPERATORS__',
                f'--threads={nvcc_threads}'
            ]
            if os.environ.get('DS_DEBUG_CUDA_BUILD', '0') == '1':
                args.append('--ptxas-options=-v')
            args += self.compute_capability_args()
        return args

    def libraries_args(self):
        if sys.platform == "win32":
            return ['cublas', 'curand']
        else:
            return []
