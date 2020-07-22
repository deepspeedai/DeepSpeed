"""
Copyright 2020 The Microsoft DeepSpeed Team

DeepSpeed library

Create a new wheel via the following command: python setup.py bdist_wheel

The wheel will be located at: dist/*.whl
"""

import os
import torch
from setuptools import setup, find_packages
from torch.utils.cpp_extension import CUDAExtension, BuildExtension

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

ext_modules = [
    CUDAExtension(
        name='deepspeed_lamb_cuda',
        sources=['csrc/lamb/fused_lamb_cuda.cpp',
                 'csrc/lamb/fused_lamb_cuda_kernel.cu'],
        include_dirs=['csrc/includes'],
        extra_compile_args={
            'cxx': [
                '-O3',
            ] + version_dependent_macros,
            'nvcc': ['-O3',
                     '--use_fast_math'] + version_dependent_macros
        }),
    CUDAExtension(name='deepspeed_transformer_cuda',
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
                  }),
    CUDAExtension(name='deepspeed_stochastic_transformer_cuda',
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
                  }),
]

setup(name='deepspeed',
      version='0.2.0',
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
