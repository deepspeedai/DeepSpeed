# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
"""
DeepSpeed library

To build wheel on Windows:
1. Install pytorch, such as pytorch 1.12 + cuda 11.6.
2. Install visual cpp build tool.
3. Include cuda toolkit.
4. Launch cmd console with Administrator privilege for creating required symlink folders.


Create a new wheel via the following command:
build_win.bat

The wheel will be located at: dist/*.whl
"""

import os
import sys
import subprocess
from setuptools import setup, find_packages
from setuptools.command import egg_info
import time
import typing

torch_available = True
try:
    import torch
except ImportError:
    torch_available = False
    print('[WARNING] Unable to import torch, pre-compiling ops will be disabled. ' \
        'Please visit https://pytorch.org/ to see how to properly install torch on your system.')

from op_builder import get_default_compute_capabilities, OpBuilder
from op_builder.all_ops import ALL_OPS
from op_builder.builder import installed_cuda_version

# Fetch rocm state.
is_rocm_pytorch = OpBuilder.is_rocm_pytorch()
rocm_version = OpBuilder.installed_rocm_version()

RED_START = '\033[31m'
RED_END = '\033[0m'
ERROR = f"{RED_START} [ERROR] {RED_END}"


def abort(msg):
    print(f"{ERROR} {msg}")
    assert False, msg


def fetch_requirements(path):
    with open(path, 'r') as fd:
        return [r.strip() for r in fd.readlines()]


def is_env_set(key):
    """
    Checks if an environment variable is set and not "".
    """
    return bool(os.environ.get(key, None))


def get_env_if_set(key, default: typing.Any = ""):
    """
    Returns an environment variable if it is set and not "",
    otherwise returns a default value. In contrast, the fallback
    parameter of os.environ.get() is skipped if the variable is set to "".
    """
    return os.environ.get(key, None) or default


install_requires = fetch_requirements('requirements/requirements.txt')
extras_require = {
    '1bit': [],  # add cupy based on cuda/rocm version
    '1bit_mpi': fetch_requirements('requirements/requirements-1bit-mpi.txt'),
    'readthedocs': fetch_requirements('requirements/requirements-readthedocs.txt'),
    'dev': fetch_requirements('requirements/requirements-dev.txt'),
    'autotuning': fetch_requirements('requirements/requirements-autotuning.txt'),
    'autotuning_ml': fetch_requirements('requirements/requirements-autotuning-ml.txt'),
    'sparse_attn': fetch_requirements('requirements/requirements-sparse_attn.txt'),
    'sparse': fetch_requirements('requirements/requirements-sparse_pruning.txt'),
    'inf': fetch_requirements('requirements/requirements-inf.txt'),
    'sd': fetch_requirements('requirements/requirements-sd.txt'),
    'triton': fetch_requirements('requirements/requirements-triton.txt'),
}

# Add specific cupy version to both onebit extension variants.
if torch_available and torch.cuda.is_available():
    cupy = None
    if is_rocm_pytorch:
        rocm_major, rocm_minor = rocm_version
        # XXX cupy support for rocm 5 is not available yet.
        if rocm_major <= 4:
            cupy = f"cupy-rocm-{rocm_major}-{rocm_minor}"
    else:
        cuda_major_ver, cuda_minor_ver = installed_cuda_version()
        if (cuda_major_ver < 11) or ((cuda_major_ver == 11) and (cuda_minor_ver < 3)):
            cupy = f"cupy-cuda{cuda_major_ver}{cuda_minor_ver}"
        else:
            cupy = f"cupy-cuda{cuda_major_ver}x"

    if cupy:
        extras_require['1bit'].append(cupy)
        extras_require['1bit_mpi'].append(cupy)

# Make an [all] extra that installs all needed dependencies.
all_extras = set()
for extra in extras_require.items():
    for req in extra[1]:
        all_extras.add(req)
extras_require['all'] = list(all_extras)

cmdclass = {}

# For any pre-installed ops force disable ninja.
if torch_available:
    from accelerator import get_accelerator
    cmdclass['build_ext'] = get_accelerator().build_extension().with_options(use_ninja=False)

if torch_available:
    TORCH_MAJOR = torch.__version__.split('.')[0]
    TORCH_MINOR = torch.__version__.split('.')[1]
else:
    TORCH_MAJOR = "0"
    TORCH_MINOR = "0"

if torch_available and not torch.cuda.is_available():
    # Fix to allow docker builds, similar to https://github.com/NVIDIA/apex/issues/486.
    print("[WARNING] Torch did not find cuda available, if cross-compiling or running with cpu only "
          "you can ignore this message. Adding compute capability for Pascal, Volta, and Turing "
          "(compute capabilities 6.0, 6.1, 6.2)")
    if not is_env_set("TORCH_CUDA_ARCH_LIST"):
        os.environ["TORCH_CUDA_ARCH_LIST"] = get_default_compute_capabilities()

ext_modules = []

# Default to pre-install kernels to false so we rely on JIT on Linux, opposite on Windows.
BUILD_OP_PLATFORM = 1 if sys.platform == "win32" else 0
BUILD_OP_DEFAULT = int(get_env_if_set('DS_BUILD_OPS', BUILD_OP_PLATFORM))
print(f"DS_BUILD_OPS={BUILD_OP_DEFAULT}")

if BUILD_OP_DEFAULT:
    assert torch_available, "Unable to pre-compile ops without torch installed. Please install torch before attempting to pre-compile ops."


def command_exists(cmd):
    if sys.platform == "win32":
        result = subprocess.Popen(f'{cmd}', stdout=subprocess.PIPE, shell=True)
        return result.wait() == 1
    else:
        result = subprocess.Popen(f'type {cmd}', stdout=subprocess.PIPE, shell=True)
        return result.wait() == 0


def op_envvar(op_name):
    assert hasattr(ALL_OPS[op_name], 'BUILD_VAR'), \
        f"{op_name} is missing BUILD_VAR field"
    return ALL_OPS[op_name].BUILD_VAR


def op_enabled(op_name):
    env_var = op_envvar(op_name)
    return int(get_env_if_set(env_var, BUILD_OP_DEFAULT))


compatible_ops = dict.fromkeys(ALL_OPS.keys(), False)
install_ops = dict.fromkeys(ALL_OPS.keys(), False)
for op_name, builder in ALL_OPS.items():
    op_compatible = builder.is_compatible()
    compatible_ops[op_name] = op_compatible
    compatible_ops["deepspeed_not_implemented"] = False

    # If op is requested but not available, throw an error.
    if op_enabled(op_name) and not op_compatible:
        env_var = op_envvar(op_name)
        if not is_env_set(env_var):
            builder.warning(f"One can disable {op_name} with {env_var}=0")
        abort(f"Unable to pre-compile {op_name}")

    # If op is compatible but install is not enabled (JIT mode).
    if is_rocm_pytorch and op_compatible and not op_enabled(op_name):
        builder.hipify_extension()

    # If op install enabled, add builder to extensions.
    if op_enabled(op_name) and op_compatible:
        assert torch_available, f"Unable to pre-compile {op_name}, please first install torch"
        install_ops[op_name] = op_enabled(op_name)
        ext_modules.append(builder.builder())

print(f'Install Ops={install_ops}')

# Write out version/git info.
git_hash_cmd = "git rev-parse --short HEAD"
git_branch_cmd = "git rev-parse --abbrev-ref HEAD"
if command_exists('git') and not is_env_set('DS_BUILD_STRING'):
    try:
        result = subprocess.check_output(git_hash_cmd, shell=True)
        git_hash = result.decode('utf-8').strip()
        result = subprocess.check_output(git_branch_cmd, shell=True)
        git_branch = result.decode('utf-8').strip()
    except subprocess.CalledProcessError:
        git_hash = "unknown"
        git_branch = "unknown"
else:
    git_hash = "unknown"
    git_branch = "unknown"


def create_dir_symlink(src, dest):
    if not os.path.islink(dest):
        if os.path.exists(dest):
            os.remove(dest)
        assert not os.path.exists(dest)
        os.symlink(src, dest)


if sys.platform == "win32":
    # This creates a symbolic links on Windows.
    # It needs Administrator privilege to create symlinks on Windows.
    create_dir_symlink('..\\..\\csrc', '.\\deepspeed\\ops\\csrc')
    create_dir_symlink('..\\..\\op_builder', '.\\deepspeed\\ops\\op_builder')
    create_dir_symlink('..\\accelerator', '.\\deepspeed\\accelerator')
    egg_info.manifest_maker.template = 'MANIFEST_win.in'

# Parse the DeepSpeed version string from version.txt.
version_str = open('version.txt', 'r').read().strip()

# Build specifiers like .devX can be added at install time. Otherwise, add the git hash.
# Example: DS_BUILD_STRING=".dev20201022" python setup.py sdist bdist_wheel.

# Building wheel for distribution, update version file.
if is_env_set('DS_BUILD_STRING'):
    # Build string env specified, probably building for distribution.
    with open('build.txt', 'w') as fd:
        fd.write(os.environ['DS_BUILD_STRING'])
    version_str += os.environ['DS_BUILD_STRING']
elif os.path.isfile('build.txt'):
    # build.txt exists, probably installing from distribution.
    with open('build.txt', 'r') as fd:
        version_str += fd.read().strip()
else:
    # None of the above, probably installing from source.
    version_str += f'+{git_hash}'

torch_version = ".".join([TORCH_MAJOR, TORCH_MINOR])
bf16_support = False
# Set cuda_version to 0.0 if cpu-only.
cuda_version = "0.0"
nccl_version = "0.0"
# Set hip_version to 0.0 if cpu-only.
hip_version = "0.0"
if torch_available and torch.version.cuda is not None:
    cuda_version = ".".join(torch.version.cuda.split('.')[:2])
    if sys.platform != "win32":
        if isinstance(torch.cuda.nccl.version(), int):
            # This will break if minor version > 9.
            nccl_version = ".".join(str(torch.cuda.nccl.version())[:2])
        else:
            nccl_version = ".".join(map(str, torch.cuda.nccl.version()[:2]))
    if hasattr(torch.cuda, 'is_bf16_supported') and torch.cuda.is_available():
        bf16_support = torch.cuda.is_bf16_supported()
if torch_available and hasattr(torch.version, 'hip') and torch.version.hip is not None:
    hip_version = ".".join(torch.version.hip.split('.')[:2])
torch_info = {
    "version": torch_version,
    "bf16_support": bf16_support,
    "cuda_version": cuda_version,
    "nccl_version": nccl_version,
    "hip_version": hip_version
}

print(f"version={version_str}, git_hash={git_hash}, git_branch={git_branch}")
with open('deepspeed/git_version_info_installed.py', 'w') as fd:
    fd.write(f"version='{version_str}'\n")
    fd.write(f"git_hash='{git_hash}'\n")
    fd.write(f"git_branch='{git_branch}'\n")
    fd.write(f"installed_ops={install_ops}\n")
    fd.write(f"compatible_ops={compatible_ops}\n")
    fd.write(f"torch_info={torch_info}\n")

print(f'install_requires={install_requires}')
print(f'compatible_ops={compatible_ops}')
print(f'ext_modules={ext_modules}')

# Parse README.md to make long_description for PyPI page.
thisdir = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(thisdir, 'README.md'), encoding='utf-8') as fin:
    readme_text = fin.read()

start_time = time.time()

setup(name='deepspeed',
      version=version_str,
      description='DeepSpeed library',
      long_description=readme_text,
      long_description_content_type='text/markdown',
      author='DeepSpeed Team',
      author_email='deepspeed-info@microsoft.com',
      url='http://deepspeed.ai',
      project_urls={
          'Documentation': 'https://deepspeed.readthedocs.io',
          'Source': 'https://github.com/microsoft/DeepSpeed',
      },
      install_requires=install_requires,
      extras_require=extras_require,
      packages=find_packages(include=['deepspeed', 'deepspeed.*']),
      include_package_data=True,
      scripts=[
          'bin/deepspeed', 'bin/deepspeed.pt', 'bin/ds', 'bin/ds_ssh', 'bin/ds_report', 'bin/ds_bench', 'bin/dsr',
          'bin/ds_elastic'
      ],
      classifiers=[
          'Programming Language :: Python :: 3.6', 'Programming Language :: Python :: 3.7',
          'Programming Language :: Python :: 3.8', 'Programming Language :: Python :: 3.9',
          'Programming Language :: Python :: 3.10'
      ],
      license='Apache Software License 2.0',
      ext_modules=ext_modules,
      cmdclass=cmdclass)

end_time = time.time()
print(f'deepspeed build time = {end_time - start_time} secs')
