---
title: "Feature Overview"
layout: single
permalink: /features/
toc: true
toc_label: "Contents"
---

## Distributed Training with Mixed Precision

### Mixed Precision Training
Enable 16-bit (FP16) training by in the `deepspeed_config` JSON.
```json
"fp16": {
    "enabled": true,
    "loss_scale": 0,
    "loss_scale_window": 1000,
    "hysteresis": 2,
    "min_loss_scale": 1
}
```

### Single-GPU, Multi-GPU, and Multi-Node Training
Easily switch between single-GPU, single-node multi-GPU, or multi-node multi-GPU
execution by specifying resources with a hostfile.
```bash
deepspeed --hostfile=<hostfile> \
	<client_entry.py> <client args> \
	--deepspeed --deepspeed_config ds_config.json
```
The script `<client_entry.py>` will execute on the resources specified in
[`<hostfile>`](/getting-started/#resource-configuration-multi-node).

## Pipeline Parallelism
DeepSpeed provides [pipeline parallelism](/tutorials/pipeline/) for memory-
and communication- efficient training. DeepSpeed supports a hybrid
combination of data, model, and pipeline parallelism and has scaled to over
[one trillion parameters using 3D parallelism]({{ site.press_release_v3 }}).
Pipeline parallelism can also improve communication efficiency and has
accelerated training by up to 7x on low-bandwidth clusters.


## Model Parallelism
### Support for Custom Model Parallelism
DeepSpeed supports all forms of model parallelism including tensor slicing
based approaches such as the
[Megatron-LM](https://github.com/NVIDIA/Megatron-LM). It does so by only
requiring the model parallelism framework to provide a *model parallelism
unit* (`mpu`) that implements a few bookkeeping functionalities:

```python
mpu.get_model_parallel_rank()
mpu.get_model_parallel_group()
mpu.get_model_parallel_world_size()

mpu.get_data_parallel_rank()
mpu.get_data_parallel_group()
mpu.get_data_parallel_world_size()
```

### Integration with Megatron-LM
DeepSpeed is fully compatible with [Megatron](https://github.com/NVIDIA/Megatron-LM).
Please see the [Megatron-LM tutorial](/tutorials/megatron/) for details.




## The Zero Redundancy Optimizer
The Zero Redundancy Optimizer ([ZeRO](https://arxiv.org/abs/1910.02054)) is at
the heart of DeepSpeed and enables large model training at a scale that is
simply not possible with model parallelism alone. When enabled, ZeRO allows
training models with over 13 billion parameters without any model parallelism,
and up to 200 billion parameter models with model parallelism on current
generation hardware.

For more details see the [ZeRO paper](https://arxiv.org/abs/1910.02054), [GPT
tutorial](/tutorials/megatron/) on integration with
DeepSpeed.

### Optimizer State and Gradient Partitioning
Optimizer State and Gradient Partitioning in ZeRO reduces the memory consumption of the
model states (optimizer states, gradients and parameters) by 8x compared to standard
data parallelism by partitioning these states across data parallel process instead of
replicating them.

### Activation Partitioning
Activation Partitioning is a memory optimization in ZeRO that can reduce the memory
consumed by activations during model parallel training (MP). In MP certain
activations maybe required by all MP processes, resulting in a replication of
activations across MP GPUs. Activation Partitioning stores these activations in a
partitioned state once they are used for computation in the forward propagation. These
activations are allgathered right before they are needed again during the backward propagation.
By storing activations in a partitioned state, ZeRO in DeepSpeed can reduce the activation
memory footprint proportional to the MP degree.

### Constant Buffer Optimization (CBO)
CBO enables high network and memory throughput while restricting memory usage to a
constant size. For memory- and network-bound operations such as normalization or
allreduce collectives, the performance depends on the size of the operand. Simply fusing
all operands into a single large operand can enable great throughput at the expense of
unnecessary memory overhead. CBO in DeepSpeed fuses smaller operands into approximately a
pre-defined sized buffer large enough to achieve great performance without the
unnecessary memory overhead.

### Contiguous Memory Optimization (CMO)
CMO reduces memory fragmentation during training, preventing out of memory errors
due to lack of contiguous memory. Memory fragmentation is a result of interleaving between
short lived and long lived memory objects. During the forward propagation activation
checkpoints are long lived but the activations that recomputed are short lived. Similarly,
during the backward computation, the activation gradients are short lived while the parameter
gradients are long lived. CMO transfers activation checkpoints and parameter gradients
to contiguous buffers preventing memory fragmentation.

## ZeRO-Offload

ZeRO-Offload pushes the boundary of the maximum model size that can be trained efficiently using minimal GPU resources, by exploiting computational and memory resources on both GPUs and their host CPUs. It allows training up to 13-billion-parameter models on a single NVIDIA V100 GPU, 10x larger than the state-of-the-art, while retaining high training throughput of over 30 teraflops per GPU.

For more details see the [ZeRO-Offload release blog]( https://www.microsoft.com/en-us/research/?p=689370&secret=iSlooB), and [tutorial](/tutorials/zero-offload/) on integration with DeepSpeed.

## Additional Memory and Bandwidth Optimizations

### Smart Gradient Accumulation
Gradient accumulation allows running larger batch size with limited memory by breaking an
effective batch into several sequential micro-batches, and averaging the parameter
gradients across these micro-batches. Furthermore, instead of averaging the gradients of
each micro-batch across all GPUs, the gradients are averaged locally during each step of
the sequence, and a single `allreduce` is done at the end of the sequence to produce the
averaged gradients for the effective batch across all GPUs. This strategy significantly
reduces the communication involved over the approach of averaging globally for each
micro-batch, specially when the number of micro-batches per effective batch is large.

### Communication Overlapping
During back propagation, DeepSpeed can overlap the communication required for averaging
parameter gradients that have already been computed with the ongoing gradient computation.
This computation-communication overlap allows DeepSpeed to achieve higher throughput even
at modest batch sizes.

## Training Features

### Simplified training API
The DeepSpeed core API consists of just a handful of methods:
* initialization: `initialize`
* training: `backward` and `step`
* argument parsing: `add_config_arguments`
* checkpointing : `load_checkpoint` and `store_checkpoint`

DeepSpeed supports most of the features described in this document, via the use of these API,
along with a `deepspeed_config` JSON file for enabling and disabling the features.
Please see the [core API doc](https://deepspeed.readthedocs.io/) for more details.

### Activation Checkpointing API

DeepSpeed's Activation Checkpointing API supports activation checkpoint partitioning,
cpu checkpointing, and contiguous memory optimizations, while also allowing layerwise
profiling. Please see the [core API doc](https://deepspeed.readthedocs.io/) for more details.


### Gradient Clipping
```json
{
  "gradient_clipping": 1.0
}
```
DeepSpeed handles gradient clipping under the hood based on the max gradient norm
specified by the user.
Please see the [core API doc](https://deepspeed.readthedocs.io/) for more details.

### Automatic loss scaling with mixed precision
DeepSpeed internally handles loss scaling for mixed precision training. The parameters
for loss scaling can be specified in the `deepspeed_config` JSON file.
Please see the [core API doc](https://deepspeed.readthedocs.io/) for more details.

## Training Optimizers

### 1-bit Adam optimizer with up to 5x less communication

DeepSpeed has an efficient implementation of a novel algorithm called 1-bit Adam.
It offers the same convergence as Adam, incurs up to 5x less communication that enables
up to 3.5x higher throughput for BERT-Large pretraining and up to 2.7x higher throughput
for SQuAD fine-tuning on bandwidth-limited clusters. For more details on usage and performance,
please refer to the detailed [tutorial](https://www.deepspeed.ai/tutorials/onebit-adam) and
[blog post](https://www.deepspeed.ai/news/2020/09/09/onebit-adam-blog-post.md), respectively.
<!-- **TODO: add paper link when it is ready ** -->

### Fused Adam optimizer and arbitrary torch.optim.Optimizer
With DeepSpeed, the user can choose to use a high performance implementation of ADAM from
NVIDIA, or any training optimizer that extends torch's `torch.optim.Optimizer` class.

### CPU-Adam: High-Performance vectorized implementation of Adam
We introduce an efficient implementation of Adam optimizer on CPU that improves the parameter-update
performance by nearly an order of magnitude. We use the AVX SIMD instructions on Intel-x86 architecture
for the CPU-Adam implementation. We support both AVX-512 and AVX-2 instruction sets. DeepSpeed uses
AVX-2 by default which can be switched to AVX-512 by setting the build flag, `DS_BUILD_AVX512` to 1 when
installing DeepSpeed. Using AVX-512, we observe 5.1x to 6.5x speedups considering the model-size between
1 to 10 billion parameters with respect to torch-adam.

### Memory bandwidth optimized FP16 Optimizer
Mixed precision training is handled by the DeepSpeed FP16 Optimizer. This optimizer not
only handles FP16 training but is also highly efficient. The performance of weight update
is primarily dominated by the memory bandwidth, and the achieved memory bandwidth is
dependent on the size of the input operands. The FP16 Optimizer is designed to maximize
the achievable memory bandwidth by merging all the parameters of the model into a single
large buffer, and applying the weight updates in a single kernel, allowing it to achieve
high memory bandwidth.

### Large Batch Training with LAMB Optimizer
<!-- **TODO: port tutorial** -->
DeepSpeed makes it easy to train with large batch sizes by enabling the LAMB Optimizer.
For more details on LAMB, see the [LAMB paper](https://arxiv.org/pdf/1904.00962.pdf).

### Memory-Efficient Training with ZeRO Optimizer
DeepSpeed can train models with up to 13 billion parameters without model parallelism, and
models with up to 200 billion parameters with 16-way model parallelism. This leap in
model size is possible through the memory efficiency achieved via the ZeRO Optimizer. For
more details see [ZeRO paper](https://arxiv.org/abs/1910.02054) .



## Training Agnostic Checkpointing
DeepSpeed can simplify checkpointing for you regardless of whether you are using data
parallel training, model parallel training, mixed-precision training, a mix of these
three, or using the zero optimizer to enable larger model sizes.
Please see the [Getting Started](/getting-started/) guide
and the [core API doc](https://deepspeed.readthedocs.io/) for more details.

## Advanced parameter search
DeepSpeed supports multiple Learning Rate Schedules to enable faster convergence for
large batch scaling.

### Learning Rate Range Test
Please refer to the [Learning Rate Range Test](/tutorials/lrrt/) tutorial.

### 1Cycle Learning Rate Schedule
Please refer to the [1Cycle Learning Rate Schedule](/tutorials/1Cycle/) tutorial.


## Simplified Data Loader
DeepSpeed abstracts away data parallelism and model parallelism from the user when it
comes to data loading. Users simply provide a PyTorch dataset, and DeepSpeed data loader
can automatically handle batch creation appropriately.

## Performance Analysis and Debugging

DeepSpeed provides a set of tools for performance analysis and debugging.

### Wall Clock Breakdown

DeepSpeed provides a detailed breakdown of the time spent
in different parts of the training.
This can be enabled by setting the following in the `deepspeed_config` file.

```json
{
  "wall_clock_breakdown": true,
}

```

###  Timing Activation Checkpoint Functions

When activation checkpointing is enabled, profiling the forward and backward time of each checkpoint function can be enabled in the `deepspeed_config` file.

```json
{
  "activation_checkpointing": {
    "profile": true
  }
}

```

### Flops Profiler

The DeepSpeed flops profiler measures the time, flops and parameters of a PyTorch model and shows which modules or layers are the bottleneck. When used with the DeepSpeed runtime, the flops profiler can be configured in the `deepspeed_config` file as follows:

```json
{
  "flops_profiler": {
    "enabled": true,
    "profile_step": 1,
    "module_depth": -1,
    "top_modules": 3,
    "detailed": true,
    }
}

```
The flops profiler can also be used as a standalone package. Please refer to the [Flops Profiler](/tutorials/flops-profiler) tutorial for more details.

## Sparse Attention
DeepSpeed offers sparse attention to support long sequences. Please refer to the [Sparse Attention](/tutorials/sparse-attention/) tutorial.

```bash
--deepspeed_sparse_attention
```

```json
"sparse_attention": {
    "mode": "fixed",
    "block": 16,
    "different_layout_per_head": true,
    "num_local_blocks": 4,
    "num_global_blocks": 1,
    "attention": "bidirectional",
    "horizontal_global_attention": false,
    "num_different_global_patterns": 4
}
```
