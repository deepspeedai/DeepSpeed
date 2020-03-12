"""
Copyright 2020 The Microsoft DeepSpeed Team

DeepSpeed library

Create a new wheel via the following command: python setup.py bdist_wheel

The wheel will be located at: dist/*.whl
"""

import os
import torch
from deepspeed import __version__ as ds_version
from setuptools import setup, find_packages
from torch.utils.cpp_extension import CUDAExtension, BuildExtension

cmdclass = {}
ext_modules = []
cmdclass['build_ext'] = BuildExtension

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

ext_modules.append(
    CUDAExtension(name='fused_lamb_cuda',
                  sources=['csrc/fused_lamb_cuda.cpp',
                           'csrc/fused_lamb_cuda_kernel.cu'],
                  extra_compile_args={
                      'cxx': [
                          '-O3',
                      ] + version_dependent_macros,
                      'nvcc': ['-O3',
                               '--use_fast_math'] + version_dependent_macros
                  }))

setup(name='deepspeed',
      version=ds_version,
      description='DeepSpeed library',
      author='DeepSpeed Team',
      author_email='deepspeed@microsoft.com',
      url='http://aka.ms/deepspeed',
      packages=find_packages(exclude=["docker",
                                      "third_party",
                                      "csrc"]),
      scripts=['bin/deepspeed',
               'bin/deepspeed.pt',
               'bin/ds',
               'bin/ds_ssh'],
      classifiers=['Programming Language :: Python :: 3.6'],
      ext_modules=ext_modules,
      cmdclass=cmdclass)
