[![Build Status](https://dev.azure.com/DeepSpeedMSFT/DeepSpeed/_apis/build/status/microsoft.DeepSpeed?branchName=master)](https://dev.azure.com/DeepSpeedMSFT/DeepSpeed/_build/latest?definitionId=1&branchName=master)
[![License MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://github.com/Microsoft/DeepSpeed/blob/master/LICENSE)

DeepSpeed is a deep learning optimization library that makes distributed training easy,
efficient, and effective.

<p align="center"><i><b>10x Larger Models</b></i></p>
<p align="center"><i><b>5x Faster Training</b></i></p>
<p align="center"><i><b>Minimal Code Change</b></i></p>

DeepSpeed can train DL models with over a hundred billion parameters on current
generation of GPU clusters, while achieving over 5x in system performance
compared to the state-of-art. Early adopters of DeepSpeed have already produced
a language model (LM) with over 17B parameters called
[Turing-NLG](https://www.microsoft.com/en-us/research/blog/turing-nlg-a-17-billion-parameter-language-model-by-microsoft),
establishing a new SOTA in the LM category.


# Table of Contents

| Section                                 | Description                                 |
| --------------------------------------- | ------------------------------------------- |
| [Why DeepSpeed?](#why-deepspeed)        |  DeepSpeed overview                         |
| [Getting Started](#getting-started)     |  DeepSpeed first steps                      |
| [Further Reading](#further-reading)     |  DeepSpeed features, tutorials, etc.        |
| [Contributing](#contributing)           |  Instructions for contributing to DeepSpeed |
| [Publications](#publications)           |  DeepSpeed publications                     |



# Why DeepSpeed?
Training advanced deep learning models is challenging. Beyond model design,
model scientists also need to set up the state-of-the-art training techniques
such as distributed training, mixed precision, gradient accumulation, and
checkpointing. Yet still, scientists may not achieve the desired system
performance and convergence rate. Large model sizes are even more challenging:
a large model easily runs out of memory with pure data parallelism and it is
difficult to use model parallelism. DeepSpeed addresses these challenges to
accelerate model development *and* training.

## Distributed, Effective, and Efficient Training with Ease
The DeepSpeed API is a lightweight wrapper on [PyTorch](https://pytorch.org/). This
means that you can use everything you love in PyTorch and without learning a new
platform. In addition, DeepSpeed manages all of the boilerplate state-of-the-art
training techniques, such as distributed training, mixed precision, gradient
accumulation, and checkpoints so that you can focus on your model development. Most
importantly, you can leverage the distinctive efficiency and effectiveness benefit of
DeepSpeed to boost speed and scale with just a few lines of code changes to your PyTorch
models.

## Speed
DeepSpeed achieves high performance and fast convergence through a combination of
efficiency optimizations on compute/communication/memory/IO and effectiveness
optimizations on advanced hyperparameter tuning and optimizers. For example:

* DeepSpeed trains BERT-large to parity in 14 hours using 64 GPUs (4 DGX-2 boxes) and in
  3.7 hours using 256 GPUs (16 DGX-2 boxes).

  **BERT-large Training Times**

  | Devices       | Source    | Training Time (hours) |
  | ------------- | --------- | ---------------------:|
  | 64 TPUs       | Google    |                    96 |
  | 64 V100 GPUs  | DeepSpeed |                **14** |
  | 256 V100 GPUs | NVIDIA    |                   3.9 |
  | 256 V100 GPUs | DeepSpeed |               **3.7** |

  <!---*Read more*: [BERT tutorial](../../Tutorials/bert_pretraining/deepspeed_bert_training.md)-->

  *BERT Tutorial*: Coming Soon

* DeepSpeed trains GPT2 (1.5 billion parameters) 3.75x faster than state-of-art, NVIDIA
  Megatron on Azure GPUs.

  *Read more*: [GPT tutorial](./docs/tutorials/MegatronGPT2Tutorial.md)



## Memory efficiency
DeepSpeed provides memory-efficient data parallelism and enables training models without
model parallelism. For example, DeepSpeed can train models with up to 6 billion parameters on
NVIDIA V100 GPUs with 32GB of device memory. In comparison, existing frameworks (e.g.,
PyTorch's Distributed Data Parallel) run out of memory with 1.5 billion parameter models.

DeepSpeed reduces the training memory footprint through a novel solution called Zero
Redundancy Optimizer (ZeRO). Unlike basic data parallelism where memory states are
replicated across data-parallel processes, ZeRO partitions model states to save
significant memory. The current implementation (stage 1 of ZeRO) reduces memory by up to
4x relative to the state-of-art. You can read more about ZeRO in our [paper](https://arxiv.org/abs/1910.02054).

With this impressive memory reduction, early adopters of DeepSpeed have already
produced  a language model (LM) with over 17B parameters called
[Turing-NLG](https://www.microsoft.com/en-us/research/blog/turing-nlg-a-17-billion-parameter-language-model-by-microsoft),
establishing a new SOTA in the LM category.


## Scalability
DeepSpeed supports efficient data parallelism, model parallelism, and their
combination. ZeRO boosts the scaling capability and efficiency further.
* DeepSpeed provides system support to run models up to 100 billion parameters,
  10x larger than the state-of-art (8 billion NVIDIA GPT, 11 billion Google T5).
* DeepSpeed can run large models more efficiently, up to 6x faster for models with
  various sizes spanning 1.5B to 100B.  More specifically, the data parallelism powered by ZeRO
  is complementary and can be combined with different types of model parallelism.  It allows
  DeepSpeed to fit models using lower degree of model parallelism and higher batch size, offering
  significant performance gains compared to using model parallelism alone.

  *Read more*: [technical report](https://arxiv.org/abs/1910.02054),
  and [GPT tutorial](./docs/tutorials/Megatron_GPT2/MegatronGPT2Tutorial.md).
  <!-- and [QANet tutorial](../../Tutorials/QANet/QANetTutorial.md). -->

![DeepSpeed-vs-Megatron](./docs/figures/DeepSpeed-vs-Megatron.png)
<p align="center">
<em>The figure depicts system throughput improvements of DeepSpeed (combining ZeRO-powered data parallelism with model parallelism of NVIDIA Megatron-LM) over using Megatron-LM alone.</em>
</p>


## Fast convergence for effectiveness
DeepSpeed supports advanced hyperparameter tuning and large batch size
optimizers such as [LAMB](https://arxiv.org/abs/1904.00962). These improve the
effectiveness of model training and reduce the number of samples required to
convergence to desired accuracy.

*Read more*: [Tuning tutorial](./docs/tutorials/1Cycle.md),
<!---
 and *BERT Tutorial*: Coming Soon.

[BERT tutorial](../../Tutorials/BingBertSquad/BingBertSquadTutorial.md),
[QANet tutorial](../../Tutorials/QANet/QANetTutorial.md)
-->


## Good Usability
Only a few lines of code changes are needed to enable a PyTorch model to use DeepSpeed and ZeRO. Compared to current model parallelism libraries, DeepSpeed does not require a code redesign or model refactoring. It also does not put limitations on model dimensions (such as number of attention heads, hidden sizes, and others), batch size, or any other training parameters. For models of up to six billion parameters, you can use ZeRO-powered data parallelism conveniently without requiring model parallelism, while in contrast, standard data parallelism will run out of memory for models with more than 1.3 billion parameters. In addition, DeepSpeed conveniently supports flexible combination of ZeRO-powered data parallelism with custom model parallelisms, such as tensor slicing of NVIDIA's Megatron-LM.


## Features

Below we provide a brief feature list, see our detailed [feature
overview](./docs/features.md) for descriptions and usage.

* [Distributed Training with Mixed Precision](./docs/features.md#distributed-training-with-mixed-precision)
    * 16-bit mixed precision
    * Single-GPU/Multi-GPU/Multi-Node
* [Model Parallelism](./docs/features.md#model-parallelism)
    * Support for Custom Model Parallelism
    * Integration with Megatron-LM
* [Memory and Bandwidth Optimizations](./docs/features.md#memory-and-bandwidth-optimizations)
    * The Zero Redundancy Optimizer (ZeRO)
    * Constant Buffer Optimization (CBO)
    * Smart Gradient Accumulation
* [Training Features](./docs/features.md#training-features)
    * Simplified training API
    * Gradient Clipping
    * Automatic loss scaling with mixed precision
* [Training Optimizers](./docs/features.md#training-optimizers)
    * Fused Adam optimizer and arbitrary `torch.optim.Optimizer`
    * Memory bandwidth optimized FP16 Optimizer
    * Large Batch Training with LAMB Optimizer
    * Memory efficient Training with ZeRO Optimizer
* [Training Agnostic Checkpointing](./docs/features.md#training-agnostic-checkpointing)
* [Advanced Parameter Search](./docs/features.md#advanced-parameter-search)
    * Learning Rate Range Test
    * 1Cycle Learning Rate Schedule
* [Simplified Data Loader](./docs/features.md#simplified-data-loader)
* [Performance Analysis and Debugging](./docs/features.md#performance-analysis-and-debugging)


# Getting Started


## Installation

* Please see our [Azure tutorial](docs/azure.md) to get started with DeepSpeed on Azure!
* If you're not on Azure, we recommend using our docker image via `docker pull deepspeed/deepspeed:latest` which contains a pre-installed version of DeepSpeed and all the necessary dependencies.
* If you want to install DeepSpeed manually, we provide an install script [install.sh](install.sh) to help install on a local machine or across an entire cluster.

## Writing DeepSpeed Models
DeepSpeed model training is accomplished using the DeepSpeed engine. The engine
can wrap any arbitrary model of type `torch.nn.module` and has a minimal set of APIs
for training and checkpointing the model. Please see the tutorials for detailed
examples.

To initialize the DeepSpeed engine:
```python
model_engine, optimizer, _, _ = deepspeed.initialize(args=cmd_args,
                                                     model=model,
                                                     model_parameters=params)
```

`deepspeed.inialize` ensures that all of the necessary setup required for
distributed data parallel or mixed precision training are done
appropriately under the hood.  In addition to wrapping the model, DeepSpeed can
construct and manage the training optimizer, data loader, and the learning rate
scheduler based on the parameters passed to `deepspeed.initialze` and the
DeepSpeed [configuration file](#deepspeed-configuration).


### Training

Once the DeepSpeed engine has been initialized, it can be used to train the
model using three simple APIs for forward propagation (`()`), backward
propagation (`backward`), and weight updates (`step`).

```python
for step, batch in enumerate(data_loader):
    #forward() method
    loss = model_engine(batch)

    #runs backpropagation
    model_engine.backward(loss)

    #weight update
    model_engine.step()
```


Under the hood, DeepSpeed automatically performs the necessary operations
required for distributed data parallel training, in mixed precision, with a
pre-defined learning rate schedule:

* **Gradient Averaging**: in distributed data parallel training, `backward`
  ensures that gradients are averaged across data parallel processes after
  training on an `train_batch_size`.

* **Loss Scaling**: in FP16/mixed precision training, the DeepSpeed
  engine automatically handles scaling the loss to avoid precision loss in the
  gradients.

* **Learning Rate Schedule**: if using DeepSpeed's learning rate
  schedule, then DeepSpeed automatically handles any updates to the learning
  rate when `step` is executed.



### Model Checkpointing
Saving and loading the training state is handled via the `save_checkpoint` and
`load_checkpoint` API in DeepSpeed which takes two arguments to uniquely
identify a checkpoint:
  * `ckpt_dir`: the directory where checkpoints will be saved.
  * `ckpt_id`: an identifier that uniquely identifies a checkpoint in the directory.
    In the following code snippet, we use the loss value as the checkpoint identifier.

```python
#load checkpoint
_, client_sd = model_engine.load_checkpoint(args.load_dir, args.ckpt_id)
step = client_sd['step']

#advance data loader to ckpt step
dataloader_to_step(data_loader, step + 1)

for step, batch in enumerate(data_loader):

    #forward() method
    loss = model_engine(batch)

    #runs backpropagation
    model_engine.backward(loss)

    #weight update
    model_engine.step()

    #save checkpoint
    if step % args.save_interval:
        client_sd['step'] = step
        ckpt_id = loss.item()
        model_engine.save_checkpoint(args.save_dir, ckpt_id, client_sd = client_sd)
```

DeepSpeed can automatically save and restore the model, optimizer, and the
learning rate scheduler states while hiding away these details from the user.
However, the user may want to save other data in addition to these that are
unique to a given model training. To support these items, `save_checkpoint`
accepts a client state dictionary `client_sd` for saving. These items can be
retrieved from `load_checkpoint` as a return argument. In the example above,
the `step` value is stored as part of the `client_sd`.


## DeepSpeed Configuration
DeepSpeed features can be enabled, disabled, or configured using a config JSON
file that should be specified as `args.deepspeed_config`. A sample config file
is shown below. For a full set of features see [core API
doc](https://microsoft.github.io/DeepSpeed/docs/htmlfiles/api/full/index.html).

```json
{
  "train_batch_size": 8,
  "gradient_accumulation_steps": 1,
  "steps_per_print": 1,
  "zero_optimization": true,
  "disable_allgather": true,
  "optimizer": {
    "type": "Adam",
    "params": {
      "lr": 0.00015,
      "max_grad_norm": 1.0
    }
  },

  "fp16": {
    "enabled": true,
    "loss_scale": 0,
    "loss_scale_window": 1000,
    "hysteresis": 2,
    "min_loss_scale": 1
  }
}
```

# Launching DeepSpeed Training
DeepSpeed installs the entry point `deepspeed` to launch distributed training.
We illustrate an example usage of DeepSpeed with the following assumptions:

1. You have already integrated DeepSpeed into your model
2. `client_entry.py` is the entry script for your model
3. `client args` is the `argparse` command line arguments
4. `ds_config.json` is the configuration file for DeepSpeed


## Resource Configuration (multi-node)
DeepSpeed configures multi-node compute resources with hostfiles that are compatible with
[OpenMPI](https://www.open-mpi.org/) and [Horovod](https://github.com/horovod/horovod).
A hostfile is a list of *hostnames* (or SSH aliases), which are machines accessible via passwordless
SSH, and *slot counts*, which specify the number of GPUs available on the system. For
example,
```
worker-1 slots=4
worker-2 slots=4
```
specifies that two machines named *worker-1* and *worker-2* each have four GPUs to use
for training.

Hostfiles are specified with the `--hostfile` command line option. If no hostfile is
specified, DeepSpeed searches for `/job/hostfile`. If no hostfile is specified or found,
DeepSpeed queries the number of GPUs on the local machine to discover the number of local
slots available.


The following command launches a PyTorch training job across all available nodes and GPUs
specified in `myhostfile`:
```bash
deepspeed <client_entry.py> <client args> \
  --deepspeed --deepspeed_config ds_config.json --hostfile=myhostfile
```

Alternatively, DeepSpeed allows you to restrict distributed training of your model to a
subset of the available nodes and GPUs. This feature is enabled through two command line
arguments: `--num_nodes` and `--num_gpus`. For example, distributed training can be
restricted to use only two nodes with the following command:
```bash
deepspeed --num_nodes=2 \
	<client_entry.py> <client args> \
	--deepspeed --deepspeed_config ds_config.json
```
You can instead include or exclude specific resources using the `--include` and
`--exclude` flags. For example, to use all available resources **except** GPU 0 on node
*worker-2* and GPUs 0 and 1 on *worker-3*:
```bash
deepspeed --exclude="worker-2:0@worker-3:0,1" \
	<client_entry.py> <client args> \
	--deepspeed --deepspeed_config ds_config.json
```
Similarly, you can use **only** GPUs 0 and 1 on *worker-2*:
```bash
deepspeed --include="worker-2:0,1" \
	<client_entry.py> <client args> \
	--deepspeed --deepspeed_config ds_config.json
```

## Resource Configuration (single-node)
In the case that we are only running on a single node (with one or more GPUs)
DeepSpeed *does not* require a hostfile as described above. If a hostfile is
not detected or passed in then DeepSpeed will query the number of GPUs on the
local machine to discover the number of slots available. The `--include` and
`--exclude` arguments work as normal, but the user should specify 'localhost'
as the hostname.


# Further Reading

| Article                                                                                        | Description                                  |
| ---------------------------------------------------------------------------------------------- | -------------------------------------------- |
| [DeepSpeed Features](./docs/features.md)                                                       |  DeepSpeed features                          |
| [DeepSpeed JSON Configuration](./docs/config_json.md)                                          |  Configuring DeepSpeed                       |
| [API Documentation]( https://microsoft.github.io/DeepSpeed/docs/htmlfiles/api/full/index.html) |  Generated DeepSpeed API documentation       |
| [CIFAR-10 Tutorial](./docs/tutorials/CIFAR-10.md)                                              |  Getting started with CIFAR-10 and DeepSpeed |
| [Megatron-LM Tutorial](./docs/tutorials/MegatronGPT2Tutorial.md)                               |  Train GPT2 with DeepSpeed and Megatron-LM   |
| [Learning Rate Range Test Tutorial](./docs/tutorials/lrrt.md)                                  |  Faster training with large learning rates   |
| [1Cycle Tutorial](./docs/tutorials/1Cycle.md)                                                  |  SOTA learning schedule in DeepSpeed         |



# Contributing
DeepSpeed welcomes your contributions! Please see our
[contributing](CONTRIBUTING.md) guide for more details on formatting, testing,
etc.

## Contributor License Agreement
This project welcomes contributions and suggestions. Most contributions require you to
agree to a Contributor License Agreement (CLA) declaring that you have the right to, and
actually do, grant us the rights to use your contribution. For details, visit
https://cla.opensource.microsoft.com.

When you submit a pull request, a CLA bot will automatically determine whether you need
to provide a CLA and decorate the PR appropriately (e.g., status check, comment). Simply
follow the instructions provided by the bot. You will only need to do this once across
all repos using our CLA.

## Code of Conduct
This project has adopted the [Microsoft Open Source Code of
Conduct](https://opensource.microsoft.com/codeofconduct/). For more information see the
[Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or contact
[opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or
comments.

# Publications
1. Samyam Rajbhandari, Jeff Rasley, Olatunji Ruwase, Yuxiong He. (2019) ZeRO: Memory Optimization Towards Training A Trillion Parameter Models. [ArXiv:1910.02054](https://arxiv.org/abs/1910.02054)
