---
title: "Training your large model with DeepSpeed"
tags: training large-model
---

## Overview

DeepSpeed has been used to train or is in the process of training some of the largest dense models in existence. These include but not limited to:

* [Megatron-Turing NLG 530B](https://www.microsoft.com/en-us/research/blog/using-deepspeed-and-megatron-to-train-megatron-turing-nlg-530b-the-worlds-largest-and-most-powerful-generative-language-model/) language model trained in collaboration with NVIDIA
* [Big Science](https://bigscience.huggingface.co/) (near 200 billion parameter) model, in collaboration with Hugging Face and hundreds of researchers around the world.
* [Turing-NLG](https://www.microsoft.com/en-us/research/blog/turing-nlg-a-17-billion-parameter-language-model-by-microsoft/) (17.2 billion parameters) trained by Microsoft

DeepSpeed offers a collection of system technologies, that has made it possible to train models at these scales. The best technology to train your large model depends on various factors such as the model architecture, batch size, inter-connect bandwidth, etc. Given the number of available choices, this can be confusing and outright daunting. This page is meant as a starting guide to help you navigate your journey towards training your large model.

## Possible ways to train a large model

At a broad level, there are two primary paths to training a large model:

* ZeRO (Zero Redundancy Optimizer) based technologies
* 3D Parallelism based technologies

**ZeRO based technologies**: In simple terms, ZeRO is a memory efficient form of data parallelism that gives you access to the aggregate GPU memory of all the GPU devices available to you, without inefficiency caused by the data replication in data parallelism. In addition, DeepSpeed also offers heterogeneous memory technologies based on ZeRO such as ZeRO-Offload and ZeRO-Infinity, which allow you to effectively leverage CPU and NVMe memory when they are available on your target systems.

Since, ZeRO is a replacement to data parallelism, it offers a seamless integration that does not require model code refactoring for existing data-parallel models. For majority of cases, ZeRO based technologies offers model scalability, training throughput efficiency without compromising ease of use.

**3D Parallelism based technologies**: 3D Parallelism refers to a combination of three different forms of parallel technologies namely tensor-slicing, pipeline-parallelism, and data parallelism (or ZeRO powered data parallelism). Combing these three forms allows for harnessing the strength of each of these technologies without the drawback of any. 3D Parallelism enables DeepSpeed to achieve excellent training throughput efficiency in the scenarios where relying on ZeRO based technologies alone might be insufficient. However, 3D parallelism requires non-trivial model code refactoring, and therefore a careful consideration is important to identify cases where 3D-Parallelism can bring non-trivial throughput benefits.

## Deciding which technology to use

**3D Parallelism for GPT-2/GPT-3 like models**: If you are attempting to train a model whose architecture resembles very closely with GPT-2 or GPT-3, then we have already done the hard work of porting 3D parallelism to a GPT-2/GPT-3 architecture-based model and have created a training pipeline that you can use to efficiently train models with hundreds of billion or even trillions of parameters. Both Megatron-Turing NLG 530B and Big Science use a variation of this code base to scale the model training. You can find the code and tutorial to get started in the [DeepSpeed-Megatron GPT-3](https://github.com/microsoft/megatron-deepspeed) repo. For more information on 3D parallelism please checkout the resources below:

[3D Parallelism Tutorial](https://www.deepspeed.ai/tutorials/pipeline/) A generic tutorial on how to port your model to use DeepSpeed 3D parallelism

[3D Parallelism Deep Dive](https://www.microsoft.com/en-us/research/blog/deepspeed-extreme-scale-model-training-for-everyone/) A Microsoft Research blog post that takes a deep dive into 3D parallelism implementation in DeepSpeed.

**ZeRO based technologies**: For most training scenarios, ZeRO offer training efficiency that is on par with 3D parallelism without requiring model code refactoring. Therefore, if you do not already have your code ported to use 3D parallelism, we suggest first trying ZeRO lines of technology to see if it fits your need. Adding ZeRO to your training pipeline with DeepSpeed is simple and does not require you to make changes to your model.  Given the trivial cost of trying out ZeRO with DeepSpeed, it is the fastest way to evaluate and decide if you should further invest in porting your model to use 3D parallelism. Enabling ZeRO with DeepSpeed also gives you access to ZeRO-Offload and ZeRO-Infinity that can enable fine tuning large models on limited GPU resources. To get started, please checkout our [ZeRO Tutorial](https://www.deepspeed.ai/tutorials/zero/).

For more in-depth information on ZeRO lines of technologies, please checkout our papers:

[ZeRO (SC20)](https://arxiv.org/pdf/1910.02054.pdf), [ZeRO Offload (ATC21) ](https://www.usenix.org/system/files/atc21-ren-jie.pdf), and [ZeRO-Infinity (SC21)](https://arxiv.org/pdf/2104.07857.pdf),

and blog posts:

[ZeRO & DeepSpeed](https://www.microsoft.com/en-us/research/blog/zero-deepspeed-new-system-optimizations-enable-training-models-with-over-100-billion-parameters/ ), [ZeRO-2 & DeepSpeed](https://www.microsoft.com/en-us/research/blog/zero-2-deepspeed-shattering-barriers-of-deep-learning-speed-scale/ ), [ZeRO-Offload](https://www.microsoft.com/en-us/research/blog/deepspeed-extreme-scale-model-training-for-everyone/ ), and [ZeRO-Infinity & DeepSpeed](https://www.microsoft.com/en-us/research/blog/zero-infinity-and-deepspeed-unlocking-unprecedented-model-scale-for-deep-learning-training/ )

## Understanding performance tradeoff between ZeRO and 3D Parallelism

The performance of ZeRO and 3D parallelism is generally on par with each other, when the batch size per GPU is not extremely small. ZeRO is a more memory efficient form of data parallelism, and the communication cost of ZeRO is quite similar to that of data parallelism itself. Therefore, for all scenarios where data parallelism works well, so will ZeRO. In fact, ZeRO enables fitting significantly larger batch sizes for large models, when compared to data parallelism due to its memory efficiency, allowing for much better throughput efficiency than data parallelism.

However, in certain scenarios the batch size may not be large enough for ZeRO to be efficient. This maybe especially true when training on thousands of GPUs or with limited network bandwidth. For example, training a GPT-3 model on 4K GPUs, and with a batch size limit of 2K will result in a batch on 0.5 per GPU, which depending on sequence length and network bandwidth might not be sufficiently large to sustain good performance using ZeRO alone.

In such scenarios, one should consider if its possible to increase the batch size to get better efficiency. However, if increasing the batch size is not an option due to convergence related concerns, then pipeline parallelism in 3D parallelism can increase the effective network bandwidth proportional to the number of pipeline stages, allowing 3D parallelism to achieve better throughput than ZeRO.
