---
layout: single
toc: true
toc_label: "Contents"
---

DeepSpeed is a deep learning optimization library that makes distributed training easy,
efficient, and effective.

<p align="center"><i><b>10x Larger Models</b></i></p>
<p align="center"><i><b>10x Faster Training</b></i></p>
<p align="center"><i><b>Minimal Code Change</b></i></p>

DeepSpeed can train DL models with over a hundred billion parameters on current
generation of GPU clusters, while achieving over 10x in system performance
compared to the state-of-art. Early adopters of DeepSpeed have already produced
a language model (LM) with over 17B parameters called
[Turing-NLG](https://www.microsoft.com/en-us/research/blog/turing-nlg-a-17-billion-parameter-language-model-by-microsoft),
establishing a new SOTA in the LM category.

DeepSpeed is an important part of Microsoft’s new
[AI at Scale](https://www.microsoft.com/en-us/research/project/ai-at-scale/)
initiative to enable next-generation AI capabilities at scale, where you can find more
information [here](https://innovation.microsoft.com/en-us/exploring-ai-at-scale).

# What's New?
{% assign news = site.posts | where: "sneak_preview", "false" %}
{% for post in news limit:5 %}
  {% if post.link %}
    {% if post.image %}
* [{{ post.date | date: "%Y/%m/%d"  }}] [ {{ post.title }} {% if post.new_post %} <span style="color:dodgerblue">**NEW!**</span> {% endif %} ![]({{ post.image }}) ]({{ post.link }})
    {% else %}
* [{{ post.date | date: "%Y/%m/%d"  }}] [{{ post.title }}]({{ post.link }}) {% if post.new_post %} <span style="color:dodgerblue">**NEW!**</span> {% endif %}
    {% endif %}
  {% else %}
* [{{ post.date | date: "%Y/%m/%d"}}] [{{ post.title }}]({{ post.url }}) {% if post.new_post %} <span style="color:dodgerblue">**NEW!**</span> {% endif %}
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

* <span style="color:dodgerblue">DeepSpeed trains BERT-large to parity in 44
  mins using 1024 V100 GPUs (64 DGX-2 boxes) and in 2.4 hours using 256 GPUs
  (16 DGX-2 boxes).</span>

  **BERT-large Training Times**

  | Devices        | Source    |        Training Time  |
  | -------------- | --------- | ---------------------:|
  | 1024 V100 GPUs | DeepSpeed |             **44** min|
  | 256 V100 GPUs  | DeepSpeed |             **2.4** hr|
  | 64 V100 GPUs   | DeepSpeed |            **8.68** hr|
  | 16 V100 GPUs   | DeepSpeed |           **33.22** hr|

  *BERT codes and tutorials will be available soon.*

* DeepSpeed trains GPT2 (1.5 billion parameters) 3.75x faster than state-of-art, NVIDIA
  Megatron on Azure GPUs.

  *Read more*: [GPT tutorial](/tutorials/megatron/)



## Memory efficiency
DeepSpeed provides memory-efficient data parallelism and enables training models without
model parallelism. For example, DeepSpeed can train models with up to 13 billion parameters on
NVIDIA V100 GPUs with 32GB of device memory. In comparison, existing frameworks (e.g.,
PyTorch's Distributed Data Parallel) run out of memory with 1.4 billion parameter models.

DeepSpeed reduces the training memory footprint through a novel solution called Zero
Redundancy Optimizer (ZeRO). Unlike basic data parallelism where memory states are
replicated across data-parallel processes, ZeRO partitions model states and gradients to save
significant memory. Furthermore, it also reduces activation memory and fragmented memory.
The current implementation (ZeRO-2) reduces memory by up to
8x relative to the state-of-art. You can read more about ZeRO in our [paper](https://arxiv.org/abs/1910.02054), and
in our blog posts related to
[ZeRO-1](https://www.microsoft.com/en-us/research/blog/zero-deepspeed-new-system-optimizations-enable-training-models-with-over-100-billion-parameters/). <!-- and [ZeRO-2](linklink). -->

With this impressive memory reduction, early adopters of DeepSpeed have already
produced  a language model (LM) with over 17B parameters called
<a href="https://www.microsoft.com/en-us/research/blog/turing-nlg-a-17-billion-parameter-language-model-by-microsoft">
<span style="color:dodgerblue">Turing-NLG</span></a>,
establishing a new SOTA in the LM category.


## Scalability
DeepSpeed supports efficient data parallelism, model parallelism, and their
combination. ZeRO boosts the scaling capability and efficiency further.
* <span style="color:dodgerblue">DeepSpeed provides system support to run models up to 170 billion parameters,
  10x larger than the state-of-art (8 billion NVIDIA GPT, 11 billion Google T5).</span>
* <span style="color:dodgerblue">DeepSpeed can run large models more efficiently, up to 10x
  faster for models with
  various sizes spanning 1.5B to 170B.</span> More specifically, the data parallelism powered by ZeRO
  is complementary and can be combined with different types of model parallelism.  It allows
  DeepSpeed to fit models using lower degree of model parallelism and higher batch size, offering
  significant performance gains compared to using model parallelism alone.

  *Read more*: [ZeRO paper](https://arxiv.org/abs/1910.02054),
  and [GPT tutorial](/tutorials/megatron).

![DeepSpeed Speedup](/assets/images/deepspeed-speedup.png)
<p align="center">
<em>The figure depicts system throughput improvements of DeepSpeed (combining ZeRO-powered data parallelism with model parallelism of NVIDIA Megatron-LM) over using Megatron-LM alone.</em>
</p>


## Fast convergence for effectiveness
DeepSpeed supports advanced hyperparameter tuning and large batch size
optimizers such as [LAMB](https://arxiv.org/abs/1904.00962). These improve the
effectiveness of model training and reduce the number of samples required to
convergence to desired accuracy.

*Read more*: [Tuning tutorial](/tutorials/1Cycle).


## Good Usability
Only a few lines of code changes are needed to enable a PyTorch model to use DeepSpeed and ZeRO. Compared to current model parallelism libraries, DeepSpeed does not require a code redesign or model refactoring. It also does not put limitations on model dimensions (such as number of attention heads, hidden sizes, and others), batch size, or any other training parameters. For models of up to 13 billion parameters, you can use ZeRO-powered data parallelism conveniently without requiring model parallelism, while in contrast, standard data parallelism will run out of memory for models with more than 1.4 billion parameters. In addition, DeepSpeed conveniently supports flexible combination of ZeRO-powered data parallelism with custom model parallelisms, such as tensor slicing of NVIDIA's Megatron-LM.


## Features

Below we provide a brief feature list, see our detailed [feature
overview](/features/) for descriptions and usage.

* [Distributed Training with Mixed Precision](/features/#distributed-training-with-mixed-precision)
    * 16-bit mixed precision
    * Single-GPU/Multi-GPU/Multi-Node
* [Model Parallelism](/features/#model-parallelism)
    * Support for Custom Model Parallelism
    * Integration with Megatron-LM
* [The Zero Redundancy Optimizer (ZeRO)](/features/#the-zero-redundancy-optimizer)
    * Optimizer State and Gradient Partitioning
    * Activation Partitioning
    * Constant Buffer Optimization
    * Contiguous Memory Optimization
* [Additional Memory and Bandwidth Optimizations](/features/#additional-memory-and-bandwidth-optimizations)
    * Smart Gradient Accumulation
    * Communication/Computation Overlap
* [Training Features](/features/#training-features)
    * Simplified training API
    * Activation Checkpointing API
    * Gradient Clipping
    * Automatic loss scaling with mixed precision
* [Training Optimizers](/features/#training-optimizers)
    * Fused Adam optimizer and arbitrary `torch.optim.Optimizer`
    * CPU-Adam: High-Performance vectorized Adam
    * Memory bandwidth optimized FP16 Optimizer
    * Large Batch Training with LAMB Optimizer
    * Memory efficient Training with ZeRO Optimizer
* [Training Agnostic Checkpointing](/features/#training-agnostic-checkpointing)
* [Advanced Parameter Search](/features/#advanced-parameter-search)
    * Learning Rate Range Test
    * 1Cycle Learning Rate Schedule
* [Simplified Data Loader](/features/#simplified-data-loader)
* [Performance Analysis and Debugging](/features/#performance-analysis-and-debugging)


# Contributing
DeepSpeed welcomes your contributions! Please see our
[contributing](/contributing/) guide for more details on formatting, testing,
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
