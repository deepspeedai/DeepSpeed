---
title: DeepSpeed Accelerator Setup Guides
tags: getting-started training accelerator
---

# Contents
- [Contents](#contents)
- [Introduction](#introduction)
- [Intel Architecture (IA) CPU](#intel-architecture-ia-cpu)
- [Intel XPU](#intel-xpu)
- [Huawei Ascend NPU](#huawei-ascend-npu)
- [Intel Gaudi](#intel-gaudi)

# Introduction
DeepSpeed supports different accelerators from different companies.   Setup steps to run DeepSpeed on certain accelerators might be different.  This guide allows user to lookup setup instructions for the accelerator family and hardware they are using.

# Intel Architecture (IA) CPU
DeepSpeed supports CPU with Intel Architecture instruction set.  It is recommended to have the CPU support at least AVX2 instruction set and recommend AMX instruction set.

DeepSpeed has been verified on the following CPU processors:
* 4th Gen Intel® Xeon® Scalarable Processors
* 5th Gen Intel® Xeon® Scalarable Processors
* 6th Gen Intel® Xeon® Scalarable Processors

## Installation steps for Intel Architecture CPU
To install DeepSpeed on Intel Architecture CPU, use the following steps:
1. Install gcc compiler
DeepSpeed requires gcc-9 or above to build kernels on Intel Architecture CPU, install gcc-9 or above.

2. Install numactl
DeepSpeed use `numactl` for fine grain CPU core allocation for load-balancing, install numactl on your system.
For example, on Ubuntu system, use the following command:
`sudo apt-get install numactl`

3. Install PyTorch
`pip install torch`

4. Install DeepSpeed
`pip install deepspeed`

## How to launch DeepSpeed on Intel Architecture CPU
DeepSpeed can launch on Intel Architecture CPU with default deepspeed command.  However, for compute intensive workloads, Intel Architecture CPU works best when each worker process runs on different set of physical CPU cores, so worker process does not compete CPU cores with each other.  To bind cores to each worker (rank), use the following command line switch for better performance.
```
deepspeed --bind_cores_to_rank <deepspeed-model-script>
```
This switch would automatically detect the number of CPU NUMA node on the host, launch the same number of workers, and bind each worker to cores/memory of a different NUMA node.  This improves performance by ensuring workers do not interfere with each other, and that all memory allocation is from local memory.

If a user wishes to have more control on the number of workers and specific cores that can be used by the workload, user can use the following command line switches.
```
deepspeed --num_accelerators <number-of-workers> --bind_cores_to_rank --bind_core_list <comma-separated-dash-range> <deepspeed-model-script>
```
For example:
```
deepspeed --num_accelerators 4 --bind_cores_to_rank --bind_core_list <0-27,32-59> inference.py
```
This would start 4 workers for the workload.  The core list range will be divided evenly between 4 workers, with worker 0 take 0-13, worker 1, take 14-27, worker 2 take 32-45, and worker 3 take 46-59.  Core 28-31,60-63 are left out because there might be some background process running on the system, leaving some idle cores will reduce performance jitting and straggler effect.

Launching DeepSpeed model on multiple CPU nodes is similar to other accelerators.  We need to specify `impi` as launcher and specify `--bind_cores_to_rank` for better core binding.  Also specify `slots` number according to number of CPU sockets in   host file.

```
# hostfile content should follow the format
# worker-1-hostname slots=<#sockets>
# worker-2-hostname slots=<#sockets>
# ...

deepspeed --hostfile=<hostfile> --bind_cores_to_rank --launcher impi --master_addr <master-ip> <deepspeed-model-script>
```

## Install with Intel Extension for PyTorch and oneCCL
Although not mandatory, Intel Extension for PyTorch and Intel oneCCL provide better optimizations for LLM models.  Intel oneCCL also provide optimization when running LLM model on multi-node.  To use DeepSpeed with Intel Extension for PyTorch and oneCCL, use the following steps:
1. Install Intel Extension for PyTorch.  This is suggested if you want to get better LLM inference performance on CPU.
`pip install intel-extension-for-pytorch`

The following steps are to install oneCCL binding for PyTorch.  This is suggested if you are running DeepSpeed on multiple CPU node, for better communication performance.   On single node with multiple CPU socket, these steps are not needed.

2. Install Intel oneCCL binding for PyTorch
`python -m pip install oneccl_bind_pt -f https://developer.intel.com/ipex-whl-stable-cpu`

3. Install Intel oneCCL, this will be used to build direct oneCCL kernels (CCLBackend kernels)
```
pip install oneccl-devel
pip install impi-devel
```
Then set the environment variables for Intel oneCCL (assuming using conda environment).
```
export CPATH=${CONDA_PREFIX}/include:$CPATH
export CCL_ROOT=${CONDA_PREFIX}
export I_MPI_ROOT=${CONDA_PREFIX}
export LD_LIBRARY_PATH=${CONDA_PREFIX}/lib/ccl/cpu:${CONDA_PREFIX}/lib/libfabric:${CONDA_PREFIX}/lib
```

## Optimize LLM inference with Intel Extension for PyTorch
Intel Extension for PyTorch compatible with DeepSpeed AutoTP tensor parallel inference.  It allows CPU inference to benefit from both DeepSpeed Automatic Tensor Parallelism, and LLM optimizations of Intel Extension for PyTorch.  To use Intel Extension for PyTorch, after calling deepspeed.init_inference, call
```
ipex_model = ipex.llm.optimize(deepspeed_model)
```
to get model optimized by Intel Extension for PyTorch.

## More examples for using DeepSpeed on Intel CPU
Refer to [LLM examples](https://github.com/intel/intel-extension-for-pytorch/tree/main/examples/cpu/llm) for more code samples of running inference with DeepSpeed on Intel CPU.


# Intel XPU
DeepSpeed XPU accelerator supports Intel® Data Center GPU Max Series.

DeepSpeed has been verified on the following GPU products:
* Intel® Data Center GPU Max 1100
* Intel® Data Center GPU Max 1550

## Installation steps for Intel XPU
To install DeepSpeed on Intel XPU, use the following steps:
1. Install oneAPI base toolkit \
The Intel® oneAPI Base Toolkit (Base Kit) is a core set of tools and libraries, including an DPC++/C++ Compiler for building Deepspeed XPU kernels like fusedAdam and CPUAdam, high performance computation libraries demanded by IPEX, etc.
For easy download, usage and more details, check [Intel oneAPI base-toolkit](https://www.intel.com/content/www/us/en/developer/tools/oneapi/base-toolkit.html).
2. Install PyTorch, Intel extension for pytorch, Intel oneCCL Bindings for PyTorch. These packages are required in `xpu_accelerator` for torch functionality and performance, also communication backend on Intel platform. The recommended installation reference:
https://intel.github.io/intel-extension-for-pytorch/index.html#installation?platform=gpu.

3. Install DeepSpeed \
`pip install deepspeed`

## How to use DeepSpeed on Intel XPU
DeepSpeed can be launched on Intel XPU with deepspeed launch command. Before that, user needs activate the oneAPI environment by: \
`source <oneAPI installed path>/setvars.sh`

To validate the XPU availability and if the XPU accelerator is correctly chosen, here is an example:
```
$ python
>>> import torch; print('torch:', torch.__version__)
torch: 2.3.0
>>> import intel_extension_for_pytorch; print('XPU available:', torch.xpu.is_available())
XPU available: True
>>> from deepspeed.accelerator import get_accelerator; print('accelerator:', get_accelerator()._name)
accelerator: xpu
```

## More examples for using DeepSpeed on Intel XPU
Refer to [LLM examples](https://github.com/intel/intel-extension-for-pytorch/tree/xpu-main/examples/gpu/llm), [Megatron-DeepSpeed training examples](https://github.com/intel/intel-extension-for-deepspeed/tree/main/examples) for more code samples of running LLM with DeepSpeed on Intel XPU.


# Huawei Ascend NPU

DeepSpeed has been verified on the following Huawei Ascend NPU products:
* Atlas 300T A2

## Installation steps for Huawei Ascend NPU

The following steps outline the process for installing DeepSpeed on an Huawei Ascend NPU:
1. Install the Huawei Ascend NPU Driver and Firmware
    <details>
    <summary>Click to expand</summary>

    Before proceeding with the installation, please download the necessary files from [Huawei Ascend NPU Driver and Firmware](https://www.hiascend.com/en/hardware/firmware-drivers/commercial?product=4&model=11).

    The following instructions below are sourced from the [Ascend Community](https://www.hiascend.com/document/detail/en/canncommercial/700/quickstart/quickstart/quickstart_18_0002.html) (refer to the [Chinese version](https://www.hiascend.com/document/detail/zh/canncommercial/700/quickstart/quickstart/quickstart_18_0002.html)):

    - Execute the following command to install the driver:
    ```
    ./Ascend-hdk-<soc_version>-npu-driver_x.x.x_linux-{arch}.run --full --install-for-all
    ```

    - Execute the following command to install the firmware:
    ```
    ./Ascend-hdk-<soc_version>-npu-firmware_x.x.x.x.X.run --full
    ```
    </details>

2. Install CANN
    <details>
    <summary>Click to expand</summary>

    Prior to installation, download the [CANN Toolkit](https://www.hiascend.com/en/software/cann/commercial).

    - Install third-party dependencies.
        - Ubuntu (The operations are the same for Debian, UOS20, and Linux.)
        ```
        apt-get install -y gcc g++ make cmake zlib1g zlib1g-dev openssl libsqlite3-dev libssl-dev libffi-dev unzip pciutils net-tools libblas-dev gfortran libblas3
        ```
        - openEuler (The operations are the same for EulerOS, CentOS, and BC-Linux.)
        ```
        yum install -y gcc gcc-c++ make cmake unzip zlib-devel libffi-devel openssl-devel pciutils net-tools sqlite-devel lapack-devel gcc-gfortran
        ```
    - Install the required Python dependencies:
    ```
    pip3 install attrs numpy decorator sympy cffi pyyaml pathlib2 psutil protobuf scipy requests absl-py wheel typing_extensions
    ```
    - Install the CANN Toolkit.
    ```
    ./Ascend-cann-toolkit_x.x.x_linux-{arch}.run --install
    ```
    </details>

3. Install PyTorch \
    `pip install torch torch_npu`

4. Install DeepSpeed \
    `pip install deepspeed`

You can view the installation results using the `ds_report` command, Here is an example:
```
--------------------------------------------------
DeepSpeed C++/CUDA extension op report
--------------------------------------------------
NOTE: Ops not installed will be just-in-time (JIT) compiled at
    runtime if needed. Op compatibility means that your system
    meet the required dependencies to JIT install the op.
--------------------------------------------------
JIT compiled ops requires ninja
ninja .................. [OKAY]
--------------------------------------------------
op name ................ installed .. compatible
--------------------------------------------------
deepspeed_not_implemented  [NO] ....... [OKAY]
async_io ............... [NO] ....... [OKAY]
cpu_adagrad ............ [NO] ....... [OKAY]
cpu_adam ............... [NO] ....... [OKAY]
cpu_lion ............... [NO] ....... [OKAY]
fused_adam ............. [NO] ....... [OKAY]
transformer_inference .. [NO] ....... [OKAY]
--------------------------------------------------
DeepSpeed general environment info:
torch install path ............... ['/root/miniconda3/envs/ds/lib/python3.10/site-packages/torch']
torch version .................... 2.2.0
deepspeed install path ........... ['/root/miniconda3/envs/ds/lib/python3.10/site-packages/deepspeed']
deepspeed info ................... 0.14.4, unknown, unknown
deepspeed wheel compiled w. ...... torch 2.2
torch_npu install path ........... ['/root/miniconda3/envs/ds/lib/python3.10/site-packages/torch_npu']
torch_npu version ................ 2.2.0
ascend_cann version .............. 8.0.RC2.alpha002
shared memory (/dev/shm) size .... 20.00 GB
```

## How to launch DeepSpeed on Huawei Ascend NPU

To validate the Huawei Ascend NPU availability and if the accelerator is correctly chosen, here is an example(Huawei Ascend NPU detection is automatic starting with DeepSpeed v0.12.6):
```
>>> import torch
>>> print('torch:',torch.__version__)
torch: 2.2.0
>>> import torch_npu
>>> print('torch_npu:',torch.npu.is_available(),",version:",torch_npu.__version__)
torch_npu: True ,version: 2.2.0
>>> from deepspeed.accelerator import get_accelerator
>>> print('accelerator:', get_accelerator()._name)
accelerator: npu
```

## Multi-card parallel training using Huawei Ascend NPU

To perform model training across multiple Huawei Ascend NPU cards using DeepSpeed, see the examples provided in [DeepSpeed Examples](https://github.com/deepspeedai/DeepSpeedExamples/blob/master/training/cifar/cifar10_deepspeed.py).

# Intel Gaudi
PyTorch models can be run on Intel® Gaudi® AI accelerator using DeepSpeed. Refer to the following user guides to start using DeepSpeed with Intel Gaudi:
* [Getting Started with DeepSpeed](https://docs.habana.ai/en/latest/PyTorch/DeepSpeed/Getting_Started_with_DeepSpeed/Getting_Started_with_DeepSpeed.html#getting-started-with-deepspeed)
* [DeepSpeed User Guide for Training](https://docs.habana.ai/en/latest/PyTorch/DeepSpeed/DeepSpeed_User_Guide/DeepSpeed_User_Guide.html#deepspeed-user-guide)
* [Optimizing Large Language Models](https://docs.habana.ai/en/latest/PyTorch/DeepSpeed/Optimizing_LLM.html#llms-opt)
* [Inference Using DeepSpeed](https://docs.habana.ai/en/latest/PyTorch/DeepSpeed/Inference_Using_DeepSpeed.html#deepspeed-inference-user-guide)
