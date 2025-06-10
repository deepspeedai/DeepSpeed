---
title: "DeepNVMe"
tags: training inference IO large-model
---
This tutorial will show how to use [DeepNVMe](https://github.com/deepspeedai/DeepSpeed/blob/master/blogs/deepspeed-gds/README.md) for data transfers between persistent storage and tensors residing in host or device memory. DeepNVMe improves the performance and efficiency of I/O operations in Deep Learning applications through powerful optimizations built on Non-Volatile Memory Express (NVMe) Solid State Drives (SSDs), Linux Asynchronous I/O (`libaio`), and NVIDIA Magnum IO<sup>TM</sup> GPUDirect® Storage (GDS).

## Requirements
Ensure your environment is properly configured to use DeepNVMe. First, you need to install DeepSpeed version >= [0.15.0](https://github.com/deepspeedai/DeepSpeed/releases/tag/v0.15.0). Next, ensure that the DeepNVMe operators are available in the DeepSpeed installation. The `async_io` operator is required for any DeepNVMe functionality, while the `gds` operator is required only for GDS functionality. You can confirm availability of each operator by inspecting the output of `ds_report` to check that compatible status is <span style="color:green">[OKAY]</span>. Below is a snippet of `ds_report` output confirming the availability of both `async_io` and `gds` operators.

![deepnvme_ops_report](/assets/images/deepnvme_ops_report.png)

If `async_io` operator is unavailable, you will need to install the appropriate `libaio` library binaries for your Linux flavor. For example, Ubuntu users will need to run `apt install libaio-dev`. In general, you should carefully inspect `ds_report` output for helpful tips such as the following:

```bash
[WARNING]  async_io requires the dev libaio .so object and headers but these were not found.
[WARNING]  async_io: please install the libaio-dev package with apt
[WARNING]  If libaio is already installed (perhaps from source), try setting the CFLAGS and LDFLAGS environment variables to where it can be found.
```

To enable `gds` operator, you will need to install NVIDIA GDS by consulting the appropriate guide for [bare-metal systems](https://docs.nvidia.com/gpudirect-storage/troubleshooting-guide/index.html) or Azure VMs (coming soon).


## Creating DeepNVMe Handles
DeepNVMe functionality can be accessed through two abstractions: `aio_handle` and `gds_handle`. The `aio_handle` is usable on both host and device tensors. while `gds_handle` works only on CUDA tensors, but is more efficient. The first step to use DeepNVMe is to create a desired handle. `aio_handle` requires `async_io` operator, while `gds_handle` requires both `async_io` and `gds` operators. The following snippets illustrate `aio_handle` and `gds_handle` creation respectively.

```python
### Create aio_handle
from deepspeed.ops.op_builder import AsyncIOBuilder
aio_handle = AsyncIOBuilder().load().aio_handle()
```

```python
### Create gds_handle
from deepspeed.ops.op_builder import GDSBuilder
gds_handle = GDSBuilder().load().gds_handle()
```

For simplicity, the above examples illustrate handle creation using default parameters. We expect that handles created with default parameters to provide good performance in most environments. However, you can see [below](#advanced-handle-creation) for advanced handle creation.

## Using DeepNVMe Handles
`aio_handle` and `gds_handle` provide identical APIs for storing tensors to files or loading tensors from files. A common feature of these APIs is that they take a tensor and a file path as arguments for the desired I/O operation. For best performance, pinned device or host tensors should be used for I/O operations (see [here](#pinned-tensors) for details). For brevity, this tutorial will use `aio_handle` for illustration, but keep in mind that `gds_handle` works similarly.

You can see the available APIs in a Python shell via tab completion on an `aio_handle` object . This is illustrated using tab completion of `h.`.

```bash
>python
Python 3.10.12 (main, Jul 29 2024, 16:56:48) [GCC 11.4.0] on linux
Type "help", "copyright", "credits" or "license" for more information.
>>> from deepspeed.ops.op_builder import AsyncIOBuilder
>>> h = AsyncIOBuilder().load().aio_handle()
>>> h.
h.async_pread(             h.free_cpu_locked_tensor(  h.get_overlap_events(      h.get_single_submit(       h.new_cpu_locked_tensor(   h.pwrite(                  h.sync_pread(              h.wait(
h.async_pwrite(            h.get_block_size(          h.get_queue_depth(         h.get_intra_op_parallelism(        h.pread(                   h.read(                    h.sync_pwrite(             h.write(
```
The APIs of interest for performing I/O operations are those named with `pread` and `pwrite` substrings. For brevity, we will focus on the file write APIs, namely `sync_pwrite`, `async_pwrite`, and `pwrite`. We will discuss only `sync_pwrite` and `async_pwrite` below because they are specializations of `pwrite`.

### Blocking File Write
`sync_pwrite` provides the standard blocking semantics of Python file write. The example below illustrates using `sync_pwrite` to store a 1GB CUDA tensor to a local NVMe file.

```bash
>>> import os
>>> os.path.isfile('/local_nvme/test_1GB.pt')
False
>>> import torch
>>> t=torch.empty(1024**3, dtype=torch.uint8).cuda()
>>> from deepspeed.ops.op_builder import AsyncIOBuilder
>>> h = AsyncIOBuilder().load().aio_handle()
>>> h.sync_pwrite(t,'/local_nvme/test_1GB.pt')
>>> os.path.isfile('/local_nvme/test_1GB.pt')
True
>>> os.path.getsize('/local_nvme/test_1GB.pt')
1073741824

```

### Non-Blocking File Write
An important DeepNVMe optimization is the non-blocking I/O semantics which enables Python threads to overlap computations with I/O operations. `async_pwrite` provides the non-blocking semantics for file writes. The Python thread can later use `wait()` to synchronize with the I/O operation. `async_write` can also be used to submit multiple back-to-back non-blocking I/O operations, of which can then be later blocked on using a single `wait()`. The example below illustrates using `async_pwrite` to store a 1GB CUDA tensor to a local NVMe file.

```bash
>>> import os
>>> os.path.isfile('/local_nvme/test_1GB.pt')
False
>>> import torch
>>> t=torch.empty(1024**3, dtype=torch.uint8).cuda()
>>> from deepspeed.ops.op_builder import AsyncIOBuilder
>>> h = AsyncIOBuilder().load().aio_handle()
>>> h.async_pwrite(t,'/local_nvme/test_1GB.pt')
>>> h.wait()
1
>>> os.path.isfile('/local_nvme/test_1GB.pt')
True
>>> os.path.getsize('/local_nvme/test_1GB.pt')
1073741824
```

<span style="color:red">Warning for non-blocking I/O operations:</span> To avoid data races and corruptions, `.wait()` must be carefully used to serialize the writing of source tensors, and the reading of destination tensors.  For example, the following update of `t` during a non-blocking file write is unsafe and could corrupt `/local_nvme/test_1GB.pt`.

```bash
>>> t=torch.empty(1024**3, dtype=torch.uint8).cuda()
>>> from deepspeed.ops.op_builder import AsyncIOBuilder
>>> h = AsyncIOBuilder().load().aio_handle()
>>> h.async_pwrite(t,'/local_nvme/test_1GB.pt')
>>> t += 1 # <--- Data race; avoid by preceding with `h.wait()`
```

Similar safety problems apply to reading the destination tensor of a non-blocking file read without `.wait()` synchronization.


### Parallel File Write
An important DeepNVMe optimization is the ability to parallelize individual I/O operations. This optimization is enabled by specifying the desired parallelism degree when constructing a DeepNVMe handle. Subsequent I/O operations with that handle are automatically parallelized over the requested number of host or device threads, as appropriate. I/O parallelism is composable with either the blocking or non-blocking I/O APIs. The example below illustrates 4-way parallelism of a file write using `async_pwrite`. Note the use of `intra_op_parallelism` argument to specify the desired parallelism degree in handle creation.

```bash
>>> import os
>>> os.path.isfile('/local_nvme/test_1GB.pt')
False
>>> import torch
>>> t=torch.empty(1024**3, dtype=torch.uint8).cuda()
>>> from deepspeed.ops.op_builder import AsyncIOBuilder
>>> h = AsyncIOBuilder().load().aio_handle(intra_op_parallelism=4)
>>> h.async_pwrite(t,'/local_nvme/test_1GB.pt')
>>> h.wait()
1
>>> os.path.isfile('/local_nvme/test_1GB.pt')
True
>>> os.path.getsize('/local_nvme/test_1GB.pt')
1073741824
```

### Pinned Tensors
A key part of DeepNVMe optimizations is using direct memory access (DMA) for I/O operations, which requires that the host or device tensor be pinned. To pin host tensors, you can use mechanisms provided by [Pytorch](https://pytorch.org/docs/stable/generated/torch.Tensor.pin_memory.html) or [DeepSpeed Accelerators](/tutorials/accelerator-abstraction-interface/#tensor-operations). The following example illustrates writing a pinned CPU tensor to a local NVMe file.

```bash
>>> import os
>>> os.path.isfile('/local_nvme/test_1GB.pt')
False
>>> import torch
>>> t=torch.empty(1024**3, dtype=torch.uint8).pin_memory()
>>> from deepspeed.ops.op_builder import AsyncIOBuilder
>>> h = AsyncIOBuilder().load().aio_handle()
>>> h.async_pwrite(t,'/local_nvme/test_1GB.pt')
>>> h.wait()
1
>>> os.path.isfile('/local_nvme/test_1GB.pt')
True
>>> os.path.getsize('/local_nvme/test_1GB.pt')
1073741824
```

On the other hand,`gds_handle` provides `new_pinned_device_tensor()` and `pin_device_tensor()` functions for pinning CUDA tensors. The following example illustrates writing a pinned CUDA tensor to a local NVMe file.

```bash
>>> import os
>>> os.path.isfile('/local_nvme/test_1GB.pt')
False
>>> import torch
>>> t=torch.empty(1024**3, dtype=torch.uint8).cuda()
>>> from deepspeed.ops.op_builder import GDSBuilder
>>> h = GDSBuilder().load().gds_handle()
>>> h.pin_device_tensor(t)
>>> h.async_pwrite(t,'/local_nvme/test_1GB.pt')
>>> h.wait()
1
>>> os.path.isfile('/local_nvme/test_1GB.pt')
True
>>> os.path.getsize('/local_nvme/test_1GB.pt')
1073741824
>>> h.unpin_device_tensor(t)
```


## Putting it together
We hope that the above material helps you to get started with DeepNVMe. You can also use the following links to see DeepNVMe usage in real-world Deep Learning applications.

1. [Parameter swapper](https://github.com/deepspeedai/DeepSpeed/blob/9b7fc5452471392b0f58844219fcfdd14a9cdc77/deepspeed/runtime/swap_tensor/partitioned_param_swapper.py#L111-L117) in [ZeRO-Inference](https://github.com/deepspeedai/DeepSpeedExamples/blob/master/inference/huggingface/zero_inference/README.md) and [ZeRO-Infinity](https://www.microsoft.com/en-us/research/blog/zero-infinity-and-deepspeed-unlocking-unprecedented-model-scale-for-deep-learning-training/).
2. [Optimizer swapper](https://github.com/deepspeedai/DeepSpeed/blob/9b7fc5452471392b0f58844219fcfdd14a9cdc77/deepspeed/runtime/swap_tensor/partitioned_optimizer_swapper.py#L36-L38) in [ZeRO-Infinity](https://www.microsoft.com/en-us/research/blog/zero-infinity-and-deepspeed-unlocking-unprecedented-model-scale-for-deep-learning-training/).
3. [Gradient swapper](https://github.com/deepspeedai/DeepSpeed/blob/9b7fc5452471392b0f58844219fcfdd14a9cdc77/deepspeed/runtime/swap_tensor/partitioned_optimizer_swapper.py#L41-L43) in [ZeRO-Infinity](https://www.microsoft.com/en-us/research/blog/zero-infinity-and-deepspeed-unlocking-unprecedented-model-scale-for-deep-learning-training/).
4. Simple file read and write [operations](https://github.com/deepspeedai/DeepSpeedExamples/blob/master/deepnvme/file_access/README.md).

<!-- 1. ZeRO-Inference: used for [parameter offloading](https://github.com/deepspeedai/DeepSpeed/blob/9b7fc5452471392b0f58844219fcfdd14a9cdc77/deepspeed/runtime/swap_tensor/partitioned_param_swapper.py#L111-L117).

2. [ZeRO-Infinity](https://www.microsoft.com/en-us/research/blog/zero-infinity-and-deepspeed-unlocking-unprecedented-model-scale-for-deep-learning-training/): used for offloading [parameters](https://github.com/deepspeedai/DeepSpeed/blob/9b7fc5452471392b0f58844219fcfdd14a9cdc77/deepspeed/runtime/swap_tensor/partitioned_param_swapper.py#L111-L117), [gradients](https://github.com/deepspeedai/DeepSpeed/blob/9b7fc5452471392b0f58844219fcfdd14a9cdc77/deepspeed/runtime/swap_tensor/partitioned_optimizer_swapper.py#L41-L43), and [optimizer](https://github.com/deepspeedai/DeepSpeed/blob/9b7fc5452471392b0f58844219fcfdd14a9cdc77/deepspeed/runtime/swap_tensor/partitioned_optimizer_swapper.py#L36-L38).
3. Simple file read and write [operations](https://github.com/deepspeedai/DeepSpeedExamples/blob/master/deepnvme/file_access/README.md).  -->


## Acknowledgements
This tutorial has been significantly improved by feedback from [Guanhua Wang](https://github.com/GuanhuaWang), [Masahiro Tanaka](https://github.com/tohtana), and [Stas Bekman](https://github.com/stas00).

## Appendix

### Advanced Handle Creation
Achieving peak I/O performance with DeepNVMe requires careful configuration of handle creation. In particular, the parameters of `aio_handle` and `gds_handle` constructors are performance-critical because they determine how efficiently DeepNVMe interacts with the underlying storage subsystem (i.e., `libaio`, GDS, PCIe, and SSD). For convenience we make it possible to create handles using default parameter values which will provide decent performance in most scenarios. However, squeezing out every available performance in your environment will likely require tuning the constructor parameters, namely `block_size`, `queue_depth`, `single_submit`, `overlap_events`, and `intra_op_parallelism`. The `aio_handle` constructor parameters and default values are illustrated below:
```bash
>>> from deepspeed.ops.op_builder import AsyncIOBuilder
>>> help(AsyncIOBuilder().load().aio_handle())
Help on aio_handle in module async_io object:

class aio_handle(pybind11_builtins.pybind11_object)
 |  Method resolution order:
 |      aio_handle
 |      pybind11_builtins.pybind11_object
 |      builtins.object
 |
 |  Methods defined here:
 |
 |  __init__(...)
 |      __init__(self: async_io.aio_handle, block_size: int = 1048576, queue_depth: int = 128, single_submit: bool = False, overlap_events: bool = False, intra_op_parallelism: int = 1) -> None
 |
 |      AIO handle constructor
```

### Performance Tuning
As discussed [earlier](#advanced-handle-creation), achieving peak DeepNVMe performance for a target workload or environment requires using optimally configured `aio_handle` or `gds_handle` handles. For configuration convenience, we provide a utility called `ds_nvme_tune` to automate the discovery of optimal DeepNVMe configurations. `ds_nvme_tune` automatically explores a user-specified or default configuration space and recommends the option that provides the best read and write performance. Below is an example usage of `ds_nvme_tune` to tune `aio_handle` data transfers between GPU memory and a local NVVMe SSD mounted on `/local_nvme`. This example used the default configuration space of `ds_nvme_tune` for tuning.

```bash
$ ds_nvme_tune --nvme_dir /local_nvme --gpu
Running DeepNVMe performance tuning on ['/local_nvme/']
Best performance (GB/sec): read =  3.69, write =  3.18
{
   "aio": {
      "single_submit": "false",
      "overlap_events": "true",
      "intra_op_parallelism": 8,
      "queue_depth": 32,
      "block_size": 1048576
   }
}
```
The above tuning was executed on a Lambda workstation equipped with two NVIDIA A6000-48GB GPUs, 252GB of DRAM, and a [CS3040 NVMe 2TB SDD](https://www.pny.com/CS3040-M2-NVMe-SSD?sku=M280CS3040-2TB-RB) with peak read and write speeds of 5.6 GB/s and 4.3 GB/s respectively. The tuning required about four and half minutes. Based on the results, one can expect to achieve read and write transfer speeds of 3.69 GB/sec and 3.18 GB/sec respectively by using an `aio_handle` configured as below.

```python
>>> from deepspeed.ops.op_builder import AsyncIOBuilder
>>> h = AsyncIOBuilder().load().aio_handle(block_size=1048576,
                                           queue_depth=32,
                                           single_submit=False,
                                           overlap_events=True,
                                           intra_op_parallelism=8)
```


The full command line options of `ds_nvme_tune` can be obtained via the normal `-h` or `--help`.
```bash
usage: ds_nvme_tune [-h] --nvme_dir NVME_DIR [NVME_DIR ...] [--sweep_config SWEEP_CONFIG] [--no_read] [--no_write] [--io_size IO_SIZE] [--gpu] [--gds] [--flush_page_cache] [--log_dir LOG_DIR] [--loops LOOPS] [--verbose]

options:
  -h, --help            show this help message and exit
  --nvme_dir NVME_DIR [NVME_DIR ...]
                        Directory in which to perform I/O tests. A writeable directory on a NVMe device.
  --sweep_config SWEEP_CONFIG
                        Performance sweep configuration json file.
  --no_read             Disable read performance measurements.
  --no_write            Disable write performance measurements.
  --io_size IO_SIZE     Number of I/O bytes to read/write for performance measurements.
  --gpu                 Test tensor transfers between GPU device and NVME device.
  --gds                 Run the sweep over NVIDIA GPUDirectStorage operator
  --flush_page_cache    Page cache will not be flushed and reported read speeds may be higher than actual ***Requires sudo access***.
  --log_dir LOG_DIR     Output directory for performance log files. Default is ./_aio_bench_logs
  --loops LOOPS         Count of operation repetitions
  --verbose             Print debugging information.
```

### DeepNVMe APIs
For convenience, we provide listing and brief descriptions of the DeepNVMe APIs.

#### General I/O APIs
The following functions are used for I/O operations with both `aio_handle` and `gds_handle`.

Function | Description |
|---|---|
async_pread | Non-blocking file read into tensor |
sync_pread | Blocking file read into tensor |
pread | File read with blocking and non-blocking options |
async_pwrite | Non-blocking file write from tensor |
sync_pwrite | Blocking file write from tensor |
pwrite | File write with blocking and non-blocking options |
wait | Wait for non-blocking I/O operations to complete |

#### GDS-specific APIs
The following functions are available only for `gds_handle`

Function | Description
|---|---|
new_pinned_device_tensor | Allocate and pin a device tensor |
free_pinned_device_tensor | Unpin and free a device tensor |
pin_device_tensor | Pin a device tensor |
unpin_device_tensor | unpin a device tensor |


#### Handle Settings APIs
The following APIs can be used to probe handle configuration.

Function | Description
|---|---|
get_queue_depth | Return queue depth setting |
get_single_submit | Return whether single_submit is enabled |
get_intra_op_parallelism | Return I/O parallelism degree |
get_block_size | Return I/O block size setting |
get_overlap_events | Return whether overlap_event is enabled |
