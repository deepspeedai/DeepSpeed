"""
Copyright 2020 The Microsoft DeepSpeed Team

DeepSpeed library

Create a new wheel via the following command: python setup.py bdist_wheel

The wheel will be located at: dist/*.whl
"""

import torch
from setuptools import setup, find_packages
from torch.utils.cpp_extension import CUDAExtension, BuildExtension

cmdclass = {}
ext_modules = []
cmdclass['build_ext'] = BuildExtension

ext_modules.append(
    CUDAExtension(name='fused_lamb_cuda',
                  sources=['csrc/fused_lamb_cuda.cpp',
                           'csrc/fused_lamb_cuda_kernel.cu'],
                  extra_compile_args={
                      'cxx': [
                          '-O3',
                      ],
                      'nvcc': ['-O3',
                               '--use_fast_math']
                  }))

setup(name='deepspeed',
      version='0.1',
      description='DeepSpeed library',
      author='DeepSpeed Team',
      author_email='deepspeed@microsoft.com',
      url='http://aka.ms/deepspeed',
      packages=find_packages(exclude=["docker",
                                      "third_party",
                                      "csrc"]),
      scripts=['bin/deepspeed',
               'bin/ds'],
      classifiers=['Programming Language :: Python :: 3.6'],
      ext_modules=ext_modules,
      cmdclass=cmdclass)
