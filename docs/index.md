---
layout: single
toc: true
toc_label: "Contents"
---

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

# What's New?
{% assign news = site.posts | where: "sneak_preview", "false" %}
{% for post in news limit:5 %}
  {% if post.link %}
  * [{{ post.title }}]({{ post.link }})
  {% else %}
  * [{{ post.title }}]({{ post.url }})
  {% endif %}
{% endfor %}


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
  and [GPT tutorial](./docs/tutorials/MegatronGPT2Tutorial.md).
  <!-- and [QANet tutorial](../../Tutorials/QANetTutorial.md). -->

![DeepSpeed-vs-Megatron](/assets/images/DeepSpeed-vs-Megatron.png)
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
overview](features) for descriptions and usage.

* [Distributed Training with Mixed Precision](features.md#distributed-training-with-mixed-precision)
    * 16-bit mixed precision
    * Single-GPU/Multi-GPU/Multi-Node
* [Model Parallelism](features.md#model-parallelism)
    * Support for Custom Model Parallelism
    * Integration with Megatron-LM
* [Memory and Bandwidth Optimizations](features.md#memory-and-bandwidth-optimizations)
    * The Zero Redundancy Optimizer (ZeRO)
    * Constant Buffer Optimization (CBO)
    * Smart Gradient Accumulation
* [Training Features](features.md#training-features)
    * Simplified training API
    * Gradient Clipping
    * Automatic loss scaling with mixed precision
* [Training Optimizers](features.md#training-optimizers)
    * Fused Adam optimizer and arbitrary `torch.optim.Optimizer`
    * Memory bandwidth optimized FP16 Optimizer
    * Large Batch Training with LAMB Optimizer
    * Memory efficient Training with ZeRO Optimizer
* [Training Agnostic Checkpointing](features.md#training-agnostic-checkpointing)
* [Advanced Parameter Search](features.md#advanced-parameter-search)
    * Learning Rate Range Test
    * 1Cycle Learning Rate Schedule
* [Simplified Data Loader](features.md#simplified-data-loader)
* [Performance Analysis and Debugging](features.md#performance-analysis-and-debugging)



# Further Reading

| Article                                                                                        | Description                                  |
| ---------------------------------------------------------------------------------------------- | -------------------------------------------- |
| [DeepSpeed Features](features.md)                                                              |  DeepSpeed features                          |
| [DeepSpeed JSON Configuration](config_json.md)                                                 |  Configuring DeepSpeed                       |
| [API Documentation](/code-docs/)                                                               |  Generated DeepSpeed API documentation       |
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
