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

DeepSpeed delivers extreme-scale model training for everyone, from data scientists training on massive supercomputers to those training on low-end clusters or even on a single GPU:
* Extreme scale: Using current generation of GPU clusters with hundreds of devices,  3D parallelism of DeepSpeed can efficiently train deep learning models with trillions of parameters.
* Extremely memory efficient: With just a single GPU, ZeRO-Offload of DeepSpeed can train models with over 10B parameters, 10x bigger than the state of arts, democratizing multi-billion-parameter model training such that many deep learning scientists can explore bigger and better models.
* Extremely long sequence length: Sparse attention of DeepSpeed powers an order-of-magnitude longer input sequence and obtains up to 6x faster execution comparing with dense transformers.
* Extremely communication efficient: 3D parallelism improves communication efficiency allows users to train multi-billion-parameter models 2–7x faster on clusters with limited network bandwidth.  1-bit Adam reduces communication volume by up to 5x while achieving similar convergence efficiency to Adam, allowing for scaling to different types of GPU clusters and networks.

Early adopters of DeepSpeed have already produced
a language model (LM) with over 17B parameters called
[Turing-NLG](https://www.microsoft.com/en-us/research/blog/turing-nlg-a-17-billion-parameter-language-model-by-microsoft),
establishing a new SOTA in the LM category.

DeepSpeed is an important part of Microsoft’s new
[AI at Scale](https://www.microsoft.com/en-us/research/project/ai-at-scale/)
initiative to enable next-generation AI capabilities at scale, where you can find more
information [here](https://innovation.microsoft.com/en-us/exploring-ai-at-scale).

# What's New?
* [2020/11/12] [Simplified install, JIT compiled ops, PyPI releases, and reduced dependencies](#installation)
* [2020/11/10] [Efficient and robust compressed training through progressive layer dropping](https://www.deepspeed.ai/news/2020/10/28/progressive-layer-dropping-news.html)
* [2020/09/10] [DeepSpeed v0.3: Extreme-scale model training for everyone]({{ site.press_release_v3 }})
  * [Powering 10x longer sequences and 6x faster execution through DeepSpeed Sparse Attention](https://www.deepspeed.ai/news/2020/09/08/sparse-attention-news.html)
  * [Training a trillion parameters with pipeline parallelism](https://www.deepspeed.ai/news/2020/09/08/pipeline-parallelism.html)
  * [Up to 5x less communication and 3.4x faster training through 1-bit Adam](https://www.deepspeed.ai/news/2020/09/08/onebit-adam-news.html)
  * [10x bigger model training on a single GPU with ZeRO-Offload](https://www.deepspeed.ai/news/2020/09/08/ZeRO-Offload.html)

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
a single GPU. In comparison, existing frameworks (e.g.,
PyTorch's Distributed Data Parallel) run out of memory with 1.4 billion parameter models.

DeepSpeed reduces the training memory footprint through a novel solution called Zero
Redundancy Optimizer (ZeRO). Unlike basic data parallelism where memory states are
replicated across data-parallel processes, ZeRO partitions model states and gradients to save
significant memory. Furthermore, it also reduces activation memory and fragmented memory.
The current implementation (ZeRO-2) reduces memory by up to
8x relative to the state-of-art. You can read more about ZeRO in our [paper](https://arxiv.org/abs/1910.02054), and
in our blog posts related to
[ZeRO-1](https://www.microsoft.com/en-us/research/blog/zero-deepspeed-new-system-optimizations-enable-training-models-with-over-100-billion-parameters/) and [ZeRO-2](https://www.microsoft.com/en-us/research/blog/zero-2-deepspeed-shattering-barriers-of-deep-learning-speed-scale/).

With this impressive memory reduction, early adopters of DeepSpeed have already
produced  a language model (LM) with over 17B parameters called
<a href="https://www.microsoft.com/en-us/research/blog/turing-nlg-a-17-billion-parameter-language-model-by-microsoft">
<span style="color:dodgerblue">Turing-NLG</span></a>,
establishing a new SOTA in the LM category.

For model scientists with limited GPU resources, ZeRO-Offload leverages both CPU and GPU memory for training large models. Using a machine with **a single GPU**, our users can run **models of up to 13 billion parameters** without running out of memory, 10x bigger than the existing approaches, while obtaining competitive throughput. This feature democratizes multi-billion-parameter model training and opens the window for many deep learning practitioners to explore bigger and better models.

## Scalability
DeepSpeed supports efficient data parallelism, model parallelism, pipeline parallelism and their
combinations, which we call 3D parallelism.
* <span style="color:dodgerblue">3D parallelism of DeepSpeed provides system support to run models with trillions of parameters, read more in our [press-release]({{ site.press_release_v3 }}) and [tutorial](/tutorials/pipeline).</span>
* <span style="color:dodgerblue">DeepSpeed can run large models more efficiently, up to 10x
  faster for models with
  various sizes spanning 1.5B to hundred billion.</span> More specifically, the data parallelism powered by ZeRO
  is complementary and can be combined with different types of model parallelism.  It allows
  DeepSpeed to fit models using lower degree of model parallelism and higher batch size, offering
  significant performance gains compared to using model parallelism alone.

  *Read more*: [ZeRO paper](https://arxiv.org/abs/1910.02054),
  and [GPT tutorial](/tutorials/megatron).

![DeepSpeed Speedup](/assets/images/deepspeed-speedup.png)
<p align="center">
<em>The figure depicts system throughput improvements of DeepSpeed (combining ZeRO-powered data parallelism with model parallelism of NVIDIA Megatron-LM) over using Megatron-LM alone.</em>
</p>

## Communication efficiency
Pipeline parallelism of DeepSpeed reduce communication volume during distributed training, which allows users to train multi-billion-parameter models 2–7x faster on clusters with limited network bandwidth.
![Low-bandwidth GPT-2 Performance](/assets/images/pp-lowbw-gpt2.png)

1-bit Adam reduces communication volume by up to 5x while achieving similar convergence efficiency to Adam, allowing for scaling to different types of GPU clusters and networks.  [Read more here](https://www.deepspeed.ai/news/2020/09/08/onebit-adam-blog-post.html).

## Supporting long sequence length
DeepSpeed offers sparse attention kernels—an instrumental technology to support long sequences of model inputs, whether for text, image, or sound. Compared with the classic dense Transformers, it powers **an order-of-magnitude longer input sequence** and obtains up to 6x faster execution with comparable accuracy. It also outperforms state-of-the-art sparse implementations with 1.5–3x faster execution. Furthermore, our sparse kernels support efficient execution of flexible sparse format and empower users to innovate on their custom sparse structures.  [Read more here](https://www.deepspeed.ai/news/2020/09/08/sparse-attention.html).


## Fast convergence for effectiveness
DeepSpeed supports advanced hyperparameter tuning and large batch size
optimizers such as [LAMB](https://arxiv.org/abs/1904.00962). These improve the
effectiveness of model training and reduce the number of samples required to
convergence to desired accuracy.

*Read more*: [Tuning tutorial](/tutorials/1Cycle).


## Good Usability
Only a few lines of code changes are needed to enable a PyTorch model to use DeepSpeed and ZeRO. Compared to current model parallelism libraries, DeepSpeed does not require a code redesign or model refactoring. It also does not put limitations on model dimensions (such as number of attention heads, hidden sizes, and others), batch size, or any other training parameters. For models of up to 13 billion parameters, you can use ZeRO-powered data parallelism conveniently without requiring model parallelism, while in contrast, standard data parallelism will run out of memory for models with more than 1.4 billion parameters. In addition, DeepSpeed conveniently supports flexible combination of ZeRO-powered data parallelism with custom model parallelisms, such as tensor slicing of NVIDIA's Megatron-LM.


## Features

Below we provide a brief feature list, see our detailed [feature overview](https://www.deepspeed.ai/features/) for descriptions and usage.

* [Distributed Training with Mixed Precision](https://www.deepspeed.ai/features/#distributed-training-with-mixed-precision)
  * 16-bit mixed precision
  * Single-GPU/Multi-GPU/Multi-Node
* [Model Parallelism](https://www.deepspeed.ai/features/#model-parallelism)
  * Support for Custom Model Parallelism
  * Integration with Megatron-LM
* [Pipeline Parallelism](https://www.deepspeed.ai/tutorials/pipeline/)
  * 3D Parallelism
* [The Zero Redundancy Optimizer (ZeRO)](https://www.deepspeed.ai/tutorials/zero/)
  * Optimizer State and Gradient Partitioning
  * Activation Partitioning
  * Constant Buffer Optimization
  * Contiguous Memory Optimization
* [ZeRO-Offload](https://www.deepspeed.ai/tutorials/zero-offload/)
  * Leverage both CPU/GPU memory for model training
  * Support 10B model training on a single GPU
* [Ultra-fast dense transformer kernels](https://www.deepspeed.ai/news/2020/05/18/bert-record.html)
* [Sparse attention](https://www.deepspeed.ai/news/2020/09/08/sparse-attention.html)
  * Memory- and compute-efficient sparse kernels
  * Support 10x long sequences than dense
  * Flexible support to different sparse structures
* [1-bit Adam](https://www.deepspeed.ai/news/2020/09/08/onebit-adam-blog-post.html)
  * Custom communication collective
  * Up to 5x communication volume saving
* [Additional Memory and Bandwidth Optimizations](https://www.deepspeed.ai/features/#additional-memory-and-bandwidth-optimizations)
  * Smart Gradient Accumulation
  * Communication/Computation Overlap
* [Training Features](https://www.deepspeed.ai/features/#training-features)
  * Simplified training API
  * Gradient Clipping
  * Automatic loss scaling with mixed precision
* [Training Optimizers](https://www.deepspeed.ai/features/#training-optimizers)
  * Fused Adam optimizer and arbitrary `torch.optim.Optimizer`
  * Memory bandwidth optimized FP16 Optimizer
  * Large Batch Training with LAMB Optimizer
  * Memory efficient Training with ZeRO Optimizer
  * CPU-Adam
* [Training Agnostic Checkpointing](https://www.deepspeed.ai/features/#training-agnostic-checkpointing)
* [Advanced Parameter Search](https://www.deepspeed.ai/features/#advanced-parameter-search)
  * Learning Rate Range Test
  * 1Cycle Learning Rate Schedule
* [Simplified Data Loader](https://www.deepspeed.ai/features/#simplified-data-loader)
* [Progressive Layer Dropping](https://www.deepspeed.ai/news/2020/10/28/progressive-layer-dropping-news.html)
  * Efficient and robust compressed training
  * Up to 2.5x convergence speedup for pre-training
* [Performance Analysis and Debugging](https://www.deepspeed.ai/features/#performance-analysis-and-debugging)


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
1. Samyam Rajbhandari, Jeff Rasley, Olatunji Ruwase, Yuxiong He. (2019) ZeRO: memory optimizations toward training trillion parameter models. [arXiv:1910.02054](https://arxiv.org/abs/1910.02054) and [In Proceedings of the International Conference for High Performance Computing, Networking, Storage and Analysis (SC '20)](https://dl.acm.org/doi/10.5555/3433701.3433727).
2. Jeff Rasley, Samyam Rajbhandari, Olatunji Ruwase, and Yuxiong He. (2020) DeepSpeed: System Optimizations Enable Training Deep Learning Models with Over 100 Billion Parameters. [In Proceedings of the 26th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining (KDD '20, Tutorial)](https://dl.acm.org/doi/10.1145/3394486.3406703).
3. Minjia Zhang, Yuxiong He. (2020) Accelerating Training of Transformer-Based Language Models with Progressive Layer Dropping. [arXiv:2010.13369](https://arxiv.org/abs/2010.13369) and [NeurIPS 2020](https://proceedings.neurips.cc/paper/2020/hash/a1140a3d0df1c81e24ae954d935e8926-Abstract.html).

# Videos
1. DeepSpeed KDD 2020 Tutorial
    1. [Overview](https://www.youtube.com/watch?v=CaseqC45DNc&list=PLa85ZdUjfWS21mgibJ2vCvLziprjpKoW0&index=29)
    2. [ZeRO + large model training](https://www.youtube.com/watch?v=y4_bCiAsIAk&list=PLa85ZdUjfWS21mgibJ2vCvLziprjpKoW0&index=28)
    3. [17B T-NLG demo](https://www.youtube.com/watch?v=9V-ZbP92drg&list=PLa85ZdUjfWS21mgibJ2vCvLziprjpKoW0&index=27)
    4. [Fastest BERT training + RScan tuning](https://www.youtube.com/watch?v=o1K-ZG9F6u0&list=PLa85ZdUjfWS21mgibJ2vCvLziprjpKoW0&index=26)
    5. DeepSpeed hands on deep dive: [part 1](https://www.youtube.com/watch?v=_NOk-mBwDYg&list=PLa85ZdUjfWS21mgibJ2vCvLziprjpKoW0&index=92), [part 2](https://www.youtube.com/watch?v=sG6_c4VXLww&list=PLa85ZdUjfWS21mgibJ2vCvLziprjpKoW0&index=94), [part 3](https://www.youtube.com/watch?v=k9yPkBTayos&list=PLa85ZdUjfWS21mgibJ2vCvLziprjpKoW0&index=93)
    6. [FAQ](https://www.youtube.com/watch?v=nsHu6vEgPew&list=PLa85ZdUjfWS21mgibJ2vCvLziprjpKoW0&index=24)
2. Microsoft Research Webinar
    * Registration is free and all videos are available on-demand.
    * [ZeRO & Fastest BERT: Increasing the scale and speed of deep learning training in DeepSpeed](https://note.microsoft.com/MSR-Webinar-DeepSpeed-Registration-On-Demand.html).
