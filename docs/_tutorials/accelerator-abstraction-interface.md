---
title: DeepSpeed Accelerator Abstraction Interface
tags: getting-started
---

# Contents
- [Contents](#contents)
- [Introduction](#introduction)
- [Write accelerator agnostic models](#write-accelerator-agnostic-models)
  - [Port accelerator runtime calls](#port-accelerator-runtime-calls)
  - [Port accelerator device name](#port-accelerator-device-name)
  - [Tensor operations](#tensor-operations)
  - [Communication backend](#communication-backend)
- [Run DeepSpeed model on different accelerators](#run-deepspeed-model-on-different-accelerators)
- [Run DeepSpeed model on CPU](#run-deepspeed-model-on-cpu)
- [Implement new accelerator extension](#implement-new-accelerator-extension)

# Introduction
The DeepSpeed Accelerator Abstraction allows user to run large language model seamlessly on various Deep Learning acceleration hardware with DeepSpeed.   It offers a set of accelerator runtime and accelerator op builder interface which can be implemented for different hardware.  This means user can write large language model code without hardware specific code.  With DeepSpeed Accelerator Abstraction, the same large language model can run on different hardware platform, without the need to rewrite model code.  This makes running large language model on different hardware easier.

This document covers three topics related to DeepSpeed Accelerator Abstraction Interface:
1. Write accelerator agnostic models using DeepSpeed Accelerator Abstraction Interface.
2. Run DeepSpeed model on different accelerators.
3. Implement new accelerator extension for DeepSpeed Accelerator Abstraction Interface.

# Write accelerator agnostic models
In this part, you will learn how to write a model that does not contain HW specific code, or how to port a model that run on a specific HW only to be accelerator agnostic.  To do this, we first import `get_accelerator` from `deepspeed.accelerator`
```
from deepspeed.accelerator import get_accelerator
```
Note: `get_accelerator()` is the entrance to DeepSpeed Accelerator Abstraction Interface
## Port accelerator runtime calls
First we need to port accelerator runtime calls.  On CUDA device, accelerator runtime call appears in the form of `torch.cuda.<interface>(...)`.   With DeepSpeed Accelerator Abstract Interface, such accelerator runtime call can be written in the form of `get_accelerator().<interface>(...)` which will be accelerator agnostic.

A typical conversion looks like the following example:

```
if torch.cuda.is_available():
    ...
```
-->
```
if get_accelerator().is_available():
    ...
```

For most `torch.cuda.<interface>(...)` call, we can literally replace `torch.cuda` with `get_accelerator()`.   However, there are some exceptions that needs attention:
1. For `torch.cuda.current_device()`, we need to know whether calling this interface is to get device index, or supply the return value as a device.   If we want to use the return value as a device string, we need to call `get_accelerator().current_device_name()`.  For example:
```
torch.empty(weight_shape, dtype=dtype, device=get_accelerator().current_device_name())
```
However, if we wish to get device index as a number, we should call `get_accelerator().current_device()`
```
local_rank = get_accelerator().current_device()
```
2. For `torch.cuda.default_generators[index]`, convert to `get_accelerator().default_generator(index)`

## Port accelerator device name
For CUDA specific device name such as `'cuda'` or `'cuda:0'`, or `'cuda:1'`, we convert them to `get_accelerator().device_name()`, `get_accelerator().device_name(0)`, and `get_accelerator().device_name(1)`.

A device name without index can be used if model need to do specific thing for certain accelerator.  We suggest to make as less as such usage only for situations can not be resolve other way.

## Tensor operations
CUDA specific tensor operations needs to be converted according to the following rules:
- When we convert a torch tensor to accelerator device such as `my_tensor.cuda()`, we use `my_tensor.to(get_accelerator().device_name())`

- When we check whether a torch tensor is on accelerator device such as `my_tensor.is_cuda`, we use `get_accelerator().on_accelerator(my_tensor)`

- When pin a tensor to GPU memory such as `my_tensor.pin_memory()`, we use `get_accelerator().pin_memory(my_tensor)`

## Communication backend
When a communication backend string is used, the interface `get_accelerator().communication_backend_name()` is used get get communication backend name. So instead of:
```
torch.distributed.init_process_group('nccl')
```
, we use:
```
torch.distributed.init_process_group(get_accelerator().communication_backend_name())
```

# Run DeepSpeed model on different accelerators
Once a model is ported with DeepSpeed Accelerator Abstraction Interface, we can run this model on different accelerators using extension to DeepSpeed.  DeepSpeed check whether certain extension is installed in the environment to decide whether to use the Accelerator backend in that extension.  For example if we wish to run model on Intel GPU, we can install _Intel Extension for DeepSpeed_ following the instruction in [link](https://github.com/intel/intel-extension-for-deepspeed/)

After the extension is installed, install DeepSpeed and run model.   The model will be running on top of DeepSpeed.   Because DeepSpeed installation is also accelerator related, it is recommended to install DeepSpeed accelerator extension before install DeepSpeed.

`CUDA_Accelerator` is the default accelerator in DeepSpeed.  If no other DeepSpeed accelerator extension is installed, `CUDA_Accelerator` will be used.

When run a model on different accelerator in a cloud environment, the recommended practice is provision environment for each accelerator in different env with tool such as _anaconda/miniconda/virtualenv_.  When run model on different Accelerator, load the env accordingly.

Note that different accelerator may have different 'flavor' of float16 or bfloat16.   So it is recommended to make the model configurable for both float16 and bfloat16, in that way model code does not need to be changed when running on different accelerators.

# Run DeepSpeed model on CPU
DeepSpeed support using CPU as accelerator.  DeepSpeed model using DeepSpeed Accelerator Abstraction Interface could run on CPU without change to model code.   DeepSpeed decide whether _Intel Extension for PyTorch_ is installed in the environment.  If this packaged is installed, DeepSpeed will use CPU as accelerator.  Otherwise CUDA device will be used as accelerator.

To run DeepSpeed model on CPU, use the following steps to prepare environment:

```
python -m pip install intel_extension_for_pytorch
python -m pip install oneccl_bind_pt -f https://developer.intel.com/ipex-whl-stable-cpu
git clone https://github.com/oneapi-src/oneCCL
cd oneCCL
mkdir build
cd build
cmake ..
make
make install
```

Before run CPU workload, we need to source oneCCL environment variables
```
source <path-to-oneCCL>/build/_install/env/setvars.sh
```

After environment is prepared, we can launch DeepSpeed inference with the following command
```
deepspeed --bind_cores_to_rank <deepspeed-model-script>
```

This command would launch number of workers equal to number of CPU sockets on the system.  Currently DeepSpeed support running inference model with AutoTP on top of CPU.  The argument `--bind_cores_to_rank` distribute CPU cores on the system evenly among workers, to allow each worker running on a dedicated set of CPU cores.

On CPU system, there might be daemon process that periodically activate which would increase variance of each worker.  One practice is leave a couple of cores for daemon process using `--bind-core-list` argument:

```
deepspeed --bind_cores_to_rank --bind_core_list 0-51,56-107 <deepspeed-model-script>
```

The command above leave 4 cores on each socket to daemon process (assume two sockets, each socket has 56 cores).

We can also set an arbitrary number of workers.  Unlike GPU, CPU cores on host can be further divided into subgroups.  When this number is not set, DeepSpeed would detect number of NUMA nodes on the system and launch one worker for each NUMA node.

```
deepspeed --num_accelerators 4 --bind_cores_to_rank <deepspeed-model-script>
```

Launching DeepSpeed model on multiple CPU nodes is similar to other accelerators.  We need to specify `impi` as launcher and specify `--bind_cores_to_rank` for better core binding.  Also specify `slots` number according to number of CPU sockets in host file.

```
# hostfile content should follow the format
# worker-1-hostname slots=<#sockets>
# worker-2-hostname slots=<#sockets>
# ...

deepspeed --hostfile=<hostfile> --bind_cores_to_rank --launcher impi --master_addr <master-ip> <deepspeed-model-script>
```

# Implement new accelerator extension
It is possible to implement a new DeepSpeed accelerator extension to support new accelerator in DeepSpeed.  An example to follow is _[Intel Extension For DeepSpeed](https://github.com/intel/intel-extension-for-deepspeed/)_.   An accelerator extension contains the following components:
1. XYZ_Accelerator(DeepSpeedAccelerator) class definition, where 'XYZ' is the accelerator name, such as 'XPU' or 'CPU'.
This class implements `class DeepSpeedAccelerator` and will be returned by `get_accelerator()` in DeepSpeed.
2. Op builders following https://github.com/intel/intel-extension-for-deepspeed/tree/main/intel_extension_for_deepspeed/op_builder.   All op builders needs to inherit `deepspeed.ops.op_builder.builder.OpBuilder` directly or indirectly.  A common practice is to implement a base op builder (SYCLOpBuilder in the case of Intel Extension for DeepSpeed) and inherit this base op builder instead.
3. Op kernels as in the following [link](https://github.com/intel/intel-extension-for-deepspeed/tree/main/intel_extension_for_deepspeed/op_builder/csrc).

Note that an extension does not have to implement all op builders under https://github.com/microsoft/DeepSpeed/tree/master/op_builder all at a time.   A missing op builder usually means certain DeepSpeed functionality cannot be used for that Accelerator, but models that does not use that functionality can still run.

When implementing op builder for an accelerator extension, one thing needs to be noted is that the op builder native code is being built by DeepSpeed jit load mechanism.  This mean the native source file being built needs to be in DeepSpeed installation directory.  However these files are defined in accelerator extension installation directory, which cannot be built by DeepSpeed directly.  To solve this, follow the example in https://github.com/intel/intel-extension-for-deepspeed/blob/main/intel_extension_for_deepspeed/op_builder/cpu_adam.py to use 'sycl_kernel_path' and 'sycl_kernel_include' (User can change 'sycl' to other prefix in their own accelerator extension) to allow native code be built during DeepSpeed jit load.

When accelerator extension is installed in the environment, it can be used by either explicit call deepspeed.accelerator.set_accelerator(XYZ_Accelerator()) following the example in https://github.com/microsoft/DeepSpeed/blob/master/accelerator/real_accelerator.py, or add an implicit detection code in get_accelerator in the same file above.
