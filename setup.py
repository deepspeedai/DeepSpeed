"""
Copyright 2020 The Microsoft DeepSpeed Team

DeepSpeed library

Create a new wheel via the following command: python setup.py bdist_wheel

The wheel will be located at: dist/*.whl
"""

import os
import torch
import subprocess
import warnings
from setuptools import setup, find_packages
from torch.utils.cpp_extension import CUDAExtension, BuildExtension, CppExtension

VERSION = "0.3.0"


def fetch_requirements(path):
    with open(path, 'r') as fd:
        return [r.strip() for r in fd.readlines()]


install_requires = fetch_requirements('requirements/requirements.txt')
dev_requires = fetch_requirements('requirements/requirements-dev.txt')
sparse_attn_requires = fetch_requirements('requirements/requirements-sparse-attn.txt')

# Build environment variables for custom builds
DS_BUILD_LAMB_MASK = 1
DS_BUILD_TRANSFORMER_MASK = 10
DS_BUILD_SPARSE_ATTN_MASK = 100

# Allow for build_cuda to turn on or off all ops
DS_BUILD_ALL_OPS = DS_BUILD_LAMB_MASK | DS_BUILD_TRANSFORMER_MASK | DS_BUILD_SPARSE_ATTN_MASK
DS_BUILD_CUDA = int(os.environ.get('DS_BUILD_CUDA', 1)) * DS_BUILD_ALL_OPS

# Set default of each op based on if build_cuda is set
OP_DEFAULT = DS_BUILD_CUDA == DS_BUILD_ALL_OPS
DS_BUILD_LAMB = int(os.environ.get('DS_BUILD_LAMB', OP_DEFAULT)) * DS_BUILD_LAMB_MASK
DS_BUILD_TRANSFORMER = int(os.environ.get('DS_BUILD_TRANSFORMER',
                                          OP_DEFAULT)) * DS_BUILD_TRANSFORMER_MASK
DS_BUILD_SPARSE_ATTN = int(os.environ.get('DS_BUILD_SPARSE_ATTN',
                                          0)) * DS_BUILD_SPARSE_ATTN_MASK

# Final effective mask is the bitwise OR of each op
BUILD_MASK = (DS_BUILD_LAMB | DS_BUILD_TRANSFORMER | DS_BUILD_SPARSE_ATTN)

install_ops = []
if BUILD_MASK & DS_BUILD_LAMB:
    install_ops.append('lamb')
if BUILD_MASK & DS_BUILD_TRANSFORMER:
    install_ops.append('transformer')
if BUILD_MASK & DS_BUILD_SPARSE_ATTN:
    install_ops.append('sparse-attn')
if len(install_ops) == 0:
    print("Building without any cuda/cpp extensions")
print(f'BUILD_MASK={BUILD_MASK}, install_ops={install_ops}')

cmdclass = {}
cmdclass['build_ext'] = BuildExtension.with_options(use_ninja=False)

TORCH_MAJOR = int(torch.__version__.split('.')[0])
TORCH_MINOR = int(torch.__version__.split('.')[1])

if not torch.cuda.is_available():
    # Fix to allow docker buils, similar to https://github.com/NVIDIA/apex/issues/486
    print(
        "[WARNING] Torch did not find cuda available, if cross-compling or running with cpu only "
        "you can ignore this message. Adding compute capability for Pascal, Volta, and Turing "
        "(compute capabilities 6.0, 6.1, 6.2)")
    if os.environ.get("TORCH_CUDA_ARCH_LIST", None) is None:
        os.environ["TORCH_CUDA_ARCH_LIST"] = "6.0;6.1;6.2;7.0;7.5"

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
version_dependent_macros = version_ge_1_1 + version_ge_1_3 + version_ge_1_5

ext_modules = []

## Lamb ##
if BUILD_MASK & DS_BUILD_LAMB:
    ext_modules.append(
        CUDAExtension(name='deepspeed.ops.lamb.fused_lamb_cuda',
                      sources=[
                          'csrc/lamb/fused_lamb_cuda.cpp',
                          'csrc/lamb/fused_lamb_cuda_kernel.cu'
                      ],
                      include_dirs=['csrc/includes'],
                      extra_compile_args={
                          'cxx': [
                              '-O3',
                          ] + version_dependent_macros,
                          'nvcc': ['-O3',
                                   '--use_fast_math'] + version_dependent_macros
                      }))

## Transformer ##
if BUILD_MASK & DS_BUILD_TRANSFORMER:
    ext_modules.append(
        CUDAExtension(name='deepspeed.ops.transformer.transformer_cuda',
                      sources=[
                          'csrc/transformer/ds_transformer_cuda.cpp',
                          'csrc/transformer/cublas_wrappers.cu',
                          'csrc/transformer/transform_kernels.cu',
                          'csrc/transformer/gelu_kernels.cu',
                          'csrc/transformer/dropout_kernels.cu',
                          'csrc/transformer/normalize_kernels.cu',
                          'csrc/transformer/softmax_kernels.cu',
                          'csrc/transformer/general_kernels.cu'
                      ],
                      include_dirs=['csrc/includes'],
                      extra_compile_args={
                          'cxx': ['-O3',
                                  '-std=c++14',
                                  '-g',
                                  '-Wno-reorder'],
                          'nvcc': [
                              '-O3',
                              '--use_fast_math',
                              '-gencode',
                              'arch=compute_61,code=compute_61',
                              '-gencode',
                              'arch=compute_70,code=compute_70',
                              '-std=c++14',
                              '-U__CUDA_NO_HALF_OPERATORS__',
                              '-U__CUDA_NO_HALF_CONVERSIONS__',
                              '-U__CUDA_NO_HALF2_OPERATORS__'
                          ]
                      }))
    ext_modules.append(
        CUDAExtension(name='deepspeed.ops.transformer.stochastic_transformer_cuda',
                      sources=[
                          'csrc/transformer/ds_transformer_cuda.cpp',
                          'csrc/transformer/cublas_wrappers.cu',
                          'csrc/transformer/transform_kernels.cu',
                          'csrc/transformer/gelu_kernels.cu',
                          'csrc/transformer/dropout_kernels.cu',
                          'csrc/transformer/normalize_kernels.cu',
                          'csrc/transformer/softmax_kernels.cu',
                          'csrc/transformer/general_kernels.cu'
                      ],
                      include_dirs=['csrc/includes'],
                      extra_compile_args={
                          'cxx': ['-O3',
                                  '-std=c++14',
                                  '-g',
                                  '-Wno-reorder'],
                          'nvcc': [
                              '-O3',
                              '--use_fast_math',
                              '-gencode',
                              'arch=compute_61,code=compute_61',
                              '-gencode',
                              'arch=compute_70,code=compute_70',
                              '-std=c++14',
                              '-U__CUDA_NO_HALF_OPERATORS__',
                              '-U__CUDA_NO_HALF_CONVERSIONS__',
                              '-U__CUDA_NO_HALF2_OPERATORS__',
                              '-D__STOCHASTIC_MODE__'
                          ]
                      }))


def command_exists(cmd):
    result = subprocess.Popen(f'type {cmd}', stdout=subprocess.PIPE, shell=True)
    return result.wait() == 0


## Sparse transformer ##
if BUILD_MASK & DS_BUILD_SPARSE_ATTN:
    # Check to see if llvm and cmake are installed since they are dependencies
    required_commands = ['llc-9', 'cmake']

    command_status = list(map(command_exists, required_commands))
    if not all(command_status):
        zipped_status = list(zip(required_commands, command_status))
        warnings.warn(
            f'Missing non-python requirements, please install the missing packages: {zipped_status}'
        )
        warnings.warn(
            'Skipping sparse attention installation due to missing required packages')
    elif TORCH_MAJOR == 1 and TORCH_MINOR >= 5:
        ext_modules.append(
            CppExtension(name='deepspeed.ops.sparse_attention.cpp_utils',
                         sources=['csrc/sparse_attention/utils.cpp'],
                         extra_compile_args={'cxx': ['-O2',
                                                     '-fopenmp']}))
        # Add sparse attention requirements
        install_requires += sparse_attn_requires
    else:
        warnings.warn('Unable to meet requirements to install sparse attention')

# Add development requirements
install_requires += dev_requires

# Write out version/git info
git_hash_cmd = "git rev-parse --short HEAD"
git_branch_cmd = "git rev-parse --abbrev-ref HEAD"
if command_exists('git'):
    result = subprocess.check_output(git_hash_cmd, shell=True)
    git_hash = result.decode('utf-8').strip()
    result = subprocess.check_output(git_branch_cmd, shell=True)
    git_branch = result.decode('utf-8').strip()
else:
    git_hash = "unknown"
    git_branch = "unknown"
print(f"version={VERSION}+{git_hash}, git_hash={git_hash}, git_branch={git_branch}")
with open('deepspeed/git_version_info.py', 'w') as fd:
    fd.write(f"version='{VERSION}+{git_hash}'\n")
    fd.write(f"git_hash='{git_hash}'\n")
    fd.write(f"git_branch='{git_branch}'\n")

print(f'install_requires={install_requires}')

setup(name='deepspeed',
      version=f"{VERSION}+{git_hash}",
      description='DeepSpeed library',
      author='DeepSpeed Team',
      author_email='deepspeed@microsoft.com',
      url='http://aka.ms/deepspeed',
      install_requires=install_requires,
      packages=find_packages(exclude=["docker",
                                      "third_party",
                                      "csrc"]),
      package_data={'deepspeed.ops.sparse_attention.trsrc': ['*.tr']},
      scripts=['bin/deepspeed',
               'bin/deepspeed.pt',
               'bin/ds',
               'bin/ds_ssh'],
      classifiers=[
          'Programming Language :: Python :: 3.6',
          'Programming Language :: Python :: 3.7',
          'Programming Language :: Python :: 3.8'
      ],
      license='MIT',
      ext_modules=ext_modules,
      cmdclass=cmdclass)
