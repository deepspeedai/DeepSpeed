"""
Copyright 2020 The Microsoft DeepSpeed Team

DeepSpeed library

To build wheel on Windows:
    1. Install pytorch, such as pytorch 1.8 + cuda 11.1
    2. Install visual cpp build tool
    3. Launch cmd console with Administrator privilege for creating required symlink folders

Create a new wheel via the following command:
    python setup.py bdist_wheel

The wheel will be located at: dist/*.whl
"""

import os
import sys
import shutil
import subprocess
import warnings
from setuptools import setup, find_packages
import time

torch_available = True
try:
    import torch
    from torch.utils.cpp_extension import BuildExtension
except ImportError:
    torch_available = False
    print('[WARNING] Unable to import torch, pre-compiling ops will be disabled. ' \
        'Please visit https://pytorch.org/ to see how to properly install torch on your system.')

from op_builder import ALL_OPS, get_default_compute_capatabilities


def fetch_requirements(path):
    with open(path, 'r') as fd:
        return [r.strip() for r in fd.readlines()]


install_requires = fetch_requirements('requirements/requirements.txt')
extras_require = {
    '1bit_adam': fetch_requirements('requirements/requirements-1bit-adam.txt'),
    'readthedocs': fetch_requirements('requirements/requirements-readthedocs.txt'),
    'dev': fetch_requirements('requirements/requirements-dev.txt'),
}

# If MPI is available add 1bit-adam requirements
if torch_available and torch.cuda.is_available():
    if shutil.which('ompi_info') or shutil.which('mpiname'):
        cupy = f"cupy-cuda{torch.version.cuda.replace('.','')[:3]}"
        extras_require['1bit_adam'].append(cupy)

# Make an [all] extra that installs all needed dependencies
all_extras = set()
for extra in extras_require.items():
    for req in extra[1]:
        all_extras.add(req)
extras_require['all'] = list(all_extras)

cmdclass = {}

# For any pre-installed ops force disable ninja
if torch_available:
    cmdclass['build_ext'] = BuildExtension.with_options(use_ninja=False)

if torch_available:
    TORCH_MAJOR = torch.__version__.split('.')[0]
    TORCH_MINOR = torch.__version__.split('.')[1]
else:
    TORCH_MAJOR = "0"
    TORCH_MINOR = "0"

if torch_available and not torch.cuda.is_available():
    # Fix to allow docker builds, similar to https://github.com/NVIDIA/apex/issues/486
    print(
        "[WARNING] Torch did not find cuda available, if cross-compiling or running with cpu only "
        "you can ignore this message. Adding compute capability for Pascal, Volta, and Turing "
        "(compute capabilities 6.0, 6.1, 6.2)")
    if os.environ.get("TORCH_CUDA_ARCH_LIST", None) is None:
        os.environ["TORCH_CUDA_ARCH_LIST"] = get_default_compute_capatabilities()

ext_modules = []

# Default to pre-install kernels to false so we rely on JIT on Linux, opposite on Windows.
BUILD_OP_PLATFORM = 1 if sys.platform == "win32" else 0
BUILD_OP_DEFAULT = int(os.environ.get('DS_BUILD_OPS', BUILD_OP_PLATFORM))
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


def op_enabled(op_name):
    assert hasattr(ALL_OPS[op_name], 'BUILD_VAR'), \
        f"{op_name} is missing BUILD_VAR field"
    env_var = ALL_OPS[op_name].BUILD_VAR
    return int(os.environ.get(env_var, BUILD_OP_DEFAULT))


install_ops = dict.fromkeys(ALL_OPS.keys(), False)
for op_name, builder in ALL_OPS.items():
    op_compatible = builder.is_compatible()

    # If op is compatible update install reqs so it can potentially build/run later
    if op_compatible:
        reqs = builder.python_requirements()
        install_requires += builder.python_requirements()

    # If op install enabled, add builder to extensions
    if op_enabled(op_name) and op_compatible:
        assert torch_available, f"Unable to pre-compile {op_name}, please first install torch"
        install_ops[op_name] = op_enabled(op_name)
        ext_modules.append(builder.builder())

compatible_ops = {op_name: op.is_compatible() for (op_name, op) in ALL_OPS.items()}

print(f'Install Ops={install_ops}')

# Write out version/git info
git_hash_cmd = "git rev-parse --short HEAD"
git_branch_cmd = "git rev-parse --abbrev-ref HEAD"
if command_exists('git') and 'DS_BUILD_STRING' not in os.environ:
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

# Parse the DeepSpeed version string from version.txt
version_str = open('version.txt', 'r').read().strip()

# Build specifiers like .devX can be added at install time. Otherwise, add the git hash.
# example: DS_BUILD_STR=".dev20201022" python setup.py sdist bdist_wheel

# Building wheel for distribution, update version file
if 'DS_BUILD_STRING' in os.environ:
    # Build string env specified, probably building for distribution
    with open('build.txt', 'w') as fd:
        fd.write(os.environ.get('DS_BUILD_STRING'))
    version_str += os.environ.get('DS_BUILD_STRING')
elif os.path.isfile('build.txt'):
    # build.txt exists, probably installing from distribution
    with open('build.txt', 'r') as fd:
        version_str += fd.read().strip()
else:
    # None of the above, probably installing from source
    version_str += f'+{git_hash}'

torch_version = ".".join([TORCH_MAJOR, TORCH_MINOR])
# Set cuda_version to 0.0 if cpu-only
cuda_version = "0.0"
if torch_available and torch.version.cuda is not None:
    cuda_version = ".".join(torch.version.cuda.split('.')[:2])
torch_info = {"version": torch_version, "cuda_version": cuda_version}

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
      author_email='deepspeed@microsoft.com',
      url='http://deepspeed.ai',
      install_requires=install_requires,
      extras_require=extras_require,
      packages=find_packages(exclude=["docker",
                                      "third_party"]),
      include_package_data=True,
      scripts=[
          'bin/deepspeed',
          'bin/deepspeed.pt',
          'bin/ds',
          'bin/ds_ssh',
          'bin/ds_report',
          'bin/ds_elastic'
      ],
      classifiers=[
          'Programming Language :: Python :: 3.6',
          'Programming Language :: Python :: 3.7',
          'Programming Language :: Python :: 3.8'
      ],
      license='MIT',
      ext_modules=ext_modules,
      cmdclass=cmdclass)

end_time = time.time()
print(f'deepspeed build time = {end_time - start_time} secs')
