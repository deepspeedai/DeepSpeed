# Feature Overview

* [Distributed Training with Mixed Precision](#distributed-training-with-mixed-precision)
    * 16-bit mixed precision
    * Single-GPU/Multi-GPU/Multi-Node
* [Model Parallelism](#model-parallelism)
    * Support for Custom Model Parallelism
    * Integration with Megatron-LM
* [Memory and Bandwidth Optimizations](#memory-and-bandwidth-optimizations)
    * The Zero Redundancy Optimizer (ZeRO)
    * Constant Buffer Optimization (CBO)
    * Smart Gradient Accumulation
* [Training Features](#training-features)
    * Simplified training API
    * Gradient Clipping
    * Automatic loss scaling with mixed precision
* [Training Optimizers](#training-optimizers)
    * Fused Adam optimizer and arbitrary `torch.optim.Optimizer`
    * Memory bandwidth optimized FP16 Optimizer
    * Large Batch Training with LAMB Optimizer
    * Memory efficient Training with ZeRO Optimizer
* [Training Agnostic Checkpointing](#training-agnostic-checkpointing)
* [Advanced Parameter Search](#advanced-parameter-search)
    * Learning Rate Range Test
    * 1Cycle Learning Rate Schedule
* [Simplified Data Loader](#simplified-data-loader)
* [Performance Analysis and Debugging](#performance-analysis-and-debugging)


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
The script `<client_entry.py>` will execute on the resources specified in `<hostfile>`.


## Model Parallelism

### Support for Custom Model Parallelism
DeepSpeed is supports all forms of model parallelism including tensor slicing based
approaches such as the [Megatron-LM](https://github.com/NVIDIA/Megatron-LM), or a
pipelined parallelism approach such as
[PipeDream](https://github.com/msr-fiddle/pipedream) or
[GPipe](https://github.com/kakaobrain/torchgpipe). It does so by only requiring the model
parallelism framework to provide a *model parallelism unit* (`mpu`) that implements a few
bookkeeping functionalities:

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
Please see the [Megatron-LM tutorial](tutorials/MegatronGPT2Tutorial.md) for details.



## Memory and Bandwidth Optimizations

### The Zero Redundancy Optimizer (ZeRO)
[ZeRO](https://arxiv.org/abs/1910.02054) is at the heart of DeepSpeed and
enables large model training at a scale that is simply not possible with model
parallelism alone. When enabled, ZeRO allows training models with
over 6 billion parameters without any model parallelism, and up to 100 billion
parameter models with model parallelism on current generation hardware.

For more details see the [ZeRO paper](https://arxiv.org/abs/1910.02054), [GPT
tutorial](tutorials/MegatronGPT2Tutorial.md) on integration with
DeepSpeed. Additional tutorials including *BERT Tutorial*: Coming Soon.
<!---[BERT
tutorial](../../Tutorials/BingBertSquad/BingBertSquadTutorial.md),
-->
### Constant Buffer Optimization (CBO)
CBO enables high network and memory throughput while restricting memory usage to a
constant size. For memory- and network-bound operations such as normalization or
allreduce collectives, the performance depends on the size of the operand. Simply fusing
all operands into a single large operand can enable great throughput at the expense of
unnecessary memory overhead. CBO in DeepSpeed fuses smaller operands into approximately a
pre-defined sized buffer large enough to achieve great performance without the
unnecessary memory overhead.

### Smart Gradient Accumulation
Gradient accumulation allows running larger batch size with limited memory by breaking an
effective batch into several sequential micro-batches, and averaging the parameter
gradients across these micro-batches. Furthermore, instead of averaging the gradients of
each micro-batch across all GPUs, the gradients are averaged locally during each step of
the sequence, and a single `allreduce` is done at the end of the sequence to produce the
averaged gradients for the effective batch across all GPUs. This strategy significantly
reduces the communication involved over the approach of averaging globally for each
micro-batch, specially when the number of micro-batches per effective batch is large.


## Training Features

### Simplified training API
The DeepSpeed core API consists of just a handful of methods:
* initialization: `initialize`
* training: `backward` and `step`
* argument parsing: `add_config_arguments`
* checkpointing : `load_checkpoint` and `store_checkpoint`

DeepSpeed supports all the features described in this document, via the use of these API,
along with a `deepspeed_config` JSON file for enabling and disabling the features.
Please see the [core API doc](https://microsoft.github.io/DeepSpeed/docs/htmlfiles/api/full/index.html) for more details.


### Gradient Clipping
DeepSpeed handles gradient clipping under the hood based on the max gradient norm
specified by the user.
Please see the [core API doc](https://microsoft.github.io/DeepSpeed/docs/htmlfiles/api/full/index.html) for more details.

### Automatic loss scaling with mixed precision
DeepSpeed internally handles loss scaling for mixed precision training. The parameters
for loss scaling can be specified in the `deepspeed_config` JSON file.
Please see the [core API doc](https://microsoft.github.io/DeepSpeed/docs/htmlfiles/api/full/index.html) for more details.

## Training Optimizers

### Fused Adam optimizer and arbitrary torch.optim.Optimizer
With DeepSpeed, the user can choose to use a high performance implementation of ADAM from
NVIDIA, or any training optimizer that extends torch's `torch.optim.Optimizer` class.

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
DeepSpeed can train models up with up to 6 billion parameters without parallelism, and
models with up to 100 billion parameters with 16-way model parallelism. This leap in
model size is possible though the memory efficiency achieved via the ZeRO Optimizer. For
more details see [ZeRO paper](https://arxiv.org/abs/1910.02054) .



## Training Agnostic Checkpointing
DeepSpeed can simplify checkpointing for you regardless of whether you are using data
parallel training, model parallel training, mixed-precision training, a mix of these
three, or using the zero optimizer to enable larger model sizes.
Please see the [Getting Started](../README.md#getting-started) guide
and the
[core API doc](https://microsoft.github.io/DeepSpeed/docs/htmlfiles/api/full/index.html) for more details.

## Advanced parameter search
DeepSpeed supports multiple Learning Rate Schedules to enable faster convergence for
large batch scaling.

### Learning Rate Range Test
Please refer to the [Learning Rate Range Test](tutorials/lrrt.md) tutorial.

### 1Cycle Learning Rate Schedule
Please refer to the [1Cycle Learning Rate Schedule](tutorials/1Cycle.md) tutorial.


## Simplified Data Loader
DeepSpeed abstracts away data parallelism and model parallelism from the user when it
comes to data loading. Users simply provide a PyTorch dataset, and DeepSpeed data loader
can automatically handle batch creation appropriately.

## Performance Analysis and Debugging
For performance debugging, DeepSpeed can give you a detailed breakdown of the time spent
in different parts of the training with by simply enabling it in the `deepspeed_config`
file.
Please see the [core API doc](https://microsoft.github.io/DeepSpeed/docs/htmlfiles/api/full/index.html) for more details.
```json
{
  "wallclock_breakdown": true
}
```
