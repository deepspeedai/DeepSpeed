import os
import time
import torch
import importlib
from pathlib import Path
import subprocess
from abc import ABC, abstractmethod

YELLOW = '\033[93m'
END = '\033[0m'
WARNING = f"{YELLOW} [WARNING] {END}"

DEFAULT_TORCH_EXTENSION_PATH = "/tmp/torch_extensions"


def assert_no_cuda_mismatch():
    import torch.utils.cpp_extension
    cuda_home = torch.utils.cpp_extension.CUDA_HOME
    assert cuda_home is not None, "CUDA_HOME does not exist, unable to compile CUDA op(s)"
    # Ensure there is not a cuda version mismatch between torch and nvcc compiler
    output = subprocess.check_output([cuda_home + "/bin/nvcc",
                                      "-V"],
                                     universal_newlines=True)
    output_split = output.split()
    release_idx = output_split.index("release")
    release = output_split[release_idx + 1].replace(',', '').split(".")
    # Ignore patch versions, only look at major + minor
    installed_cuda_version = ".".join(release[:2])
    torch_cuda_version = ".".join(torch.version.cuda.split('.')[:2])
    # This is a show-stopping error, should probably not proceed past this
    if installed_cuda_version != torch_cuda_version:
        raise Exception(
            f"Installed CUDA version {installed_cuda_version} does not match the "
            f"version torch was compiled with {torch.version.cuda}, unable to compile "
            "cuda/cpp extensions without a matching cuda version.")


def assert_torch_info(torch_info):
    install_torch_version = torch_info['version']
    install_cuda_version = torch_info['cuda_version']

    current_cuda_version = ".".join(torch.version.cuda.split('.')[:2])
    current_torch_version = ".".join(torch.__version__.split('.')[:2])

    if install_cuda_version != current_cuda_version or install_torch_version != current_torch_version:
        raise RuntimeError(
            "PyTorch and CUDA version mismatch! DeepSpeed ops were compiled and installed "
            "with a different version than what is being used at runtime. Please re-install "
            f"DeepSpeed or switch torch versions. DeepSpeed install versions: "
            f"torch={install_torch_version}, cuda={install_cuda_version}, runtime versions:"
            f"torch={current_torch_version}, cuda={current_cuda_version}")


class OpBuilder(ABC):
    def __init__(self, name):
        self.name = name
        self.jit_mode = False

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

    def include_paths(self):
        '''
        Returns list of include paths, relative to root of deepspeed package (i.e., DeepSpeed/deepspeed)
        '''
        return []

    def nvcc_args(self):
        '''
        Returns optional list of compiler flags to forward to nvcc when building CUDA sources
        '''
        return []

    def cxx_args(self):
        '''
        Returns optional list of compiler flags to forward to the build
        '''
        return []

    def is_compatible(self):
        '''
        Check if all non-python dependencies are satisfied to build this op
        '''
        return True

    def python_requirements(self):
        '''
        Override if op wants to define special dependencies, otherwise will
        take self.name and load requirements-<op-name>.txt if it exists.
        '''
        path = f'requirements/requirements-{self.name}.txt'
        requirements = []
        if os.path.isfile(path):
            with open(path, 'r') as fd:
                requirements = [r.strip() for r in fd.readlines()]
        return requirements

    def command_exists(self, cmd):
        if '|' in cmd:
            cmds = cmd.split("|")
        else:
            cmds = [cmd]
        valid = False
        for cmd in cmds:
            result = subprocess.Popen(f'type {cmd}', stdout=subprocess.PIPE, shell=True)
            valid = valid or result.wait() == 0

        if not valid and len(cmds) > 1:
            print(
                f"{WARNING} {self.name} requires one of the following commands '{cmds}', but it does not exist!"
            )
        elif not valid and len(cmds) == 1:
            print(
                f"{WARNING} {self.name} requires the '{cmd}' command, but it does not exist!"
            )
        return valid

    def warning(self, msg):
        print(f"{WARNING} {msg}")

    def deepspeed_src_path(self, code_path):
        if os.path.isabs(code_path):
            return code_path
        else:
            return os.path.join(Path(__file__).parent.parent.absolute(), code_path)

    def builder(self):
        from torch.utils.cpp_extension import CppExtension
        return CppExtension(name=self.absolute_name(),
                            sources=self.sources(),
                            include_dirs=self.include_paths(),
                            extra_compile_args={'cxx': self.cxx_args()})

    def load(self, verbose=True):
        from ...git_version_info import installed_ops, torch_info
        if installed_ops[self.name]:
            # Ensure the op we're about to load was compiled with the same
            # torch/cuda versions we are currently using at runtime.
            if isinstance(self, CUDAOpBuilder):
                assert_torch_info(torch_info)

            return importlib.import_module(self.absolute_name())
        else:
            return self.jit_load(verbose)

    def jit_load(self, verbose=True):
        if not self.is_compatible():
            raise RuntimeError(
                f"Unable to JIT load the {self.name} op due to it not being compatible due to hardware/software issue."
            )
        try:
            import ninja
        except ImportError:
            raise RuntimeError(
                f"Unable to JIT load the {self.name} op due to ninja not being installed."
            )

        if isinstance(self, CUDAOpBuilder):
            assert_no_cuda_mismatch()

        self.jit_mode = True
        from torch.utils.cpp_extension import load

        # Ensure directory exists to prevent race condition in some cases
        ext_path = os.path.join(
            os.environ.get('TORCH_EXTENSIONS_DIR',
                           DEFAULT_TORCH_EXTENSION_PATH),
            self.name)
        os.makedirs(ext_path, exist_ok=True)

        start_build = time.time()
        op_module = load(
            name=self.name,
            sources=[self.deepspeed_src_path(path) for path in self.sources()],
            extra_include_paths=[
                self.deepspeed_src_path(path) for path in self.include_paths()
            ],
            extra_cflags=self.cxx_args(),
            extra_cuda_cflags=self.nvcc_args(),
            verbose=verbose)
        build_duration = time.time() - start_build
        if verbose:
            print(f"Time to load {self.name} op: {build_duration} seconds")
        return op_module


class CUDAOpBuilder(OpBuilder):
    def compute_capability_args(self, cross_compile_archs=['60', '61', '70']):
        args = []
        if self.jit_mode:
            # Compile for underlying architecture since we know it at runtime
            CC_MAJOR, CC_MINOR = torch.cuda.get_device_capability()
            compute_capability = f"{CC_MAJOR}{CC_MINOR}"
            args.append('-gencode')
            args.append(
                f'arch=compute_{compute_capability},code=compute_{compute_capability}')
        else:
            # Cross-compile mode, compile for various architectures
            for compute_capability in cross_compile_archs:
                args.append('-gencode')
                args.append(
                    f'arch=compute_{compute_capability},code=compute_{compute_capability}'
                )
        return args

    def version_dependent_macros(self):
        # Fix from apex that might be relevant for us as well, related to https://github.com/NVIDIA/apex/issues/456
        TORCH_MAJOR = int(torch.__version__.split('.')[0])
        TORCH_MINOR = int(torch.__version__.split('.')[1])
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

    def is_compatible(self):
        return super().is_compatible()

    def builder(self):
        from torch.utils.cpp_extension import CUDAExtension
        assert_no_cuda_mismatch()
        return CUDAExtension(name=self.absolute_name(),
                             sources=self.sources(),
                             include_dirs=self.include_paths(),
                             extra_compile_args={
                                 'cxx': self.cxx_args(),
                                 'nvcc': self.nvcc_args()
                             })
