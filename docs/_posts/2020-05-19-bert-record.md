---
title: "The Fastest and Most Efficient BERT Training through Optimized Transformer Kernels"
excerpt: ""
date: 2020-05-19 00:00:00
toc: false
tags: training English
---

We introduce new technology to accelerate single GPU performance via kernel
optimizations. These optimizations not only create a strong foundation for
scaling out large models, but also improve the single GPU performance of
highly tuned and moderately sized models like BERT by more than 30%, reaching
a staggering performance of 66 teraflops per V100 GPU, which is 52% of the
hardware peak. **Using optimized transformer kernels as the building block,
DeepSpeed achieves the fastest BERT training record: 44 minutes on 1,024
NVIDIA V100 GPUs**, compared with the best published result of 67 minutes on
the same number and generation of GPUs.

* Brief overview, see our [press release](https://www.microsoft.com/en-us/research/blog/ZeRO-2-deepspeed-shattering-barriers-of-deep-learning-speed-scale/).
* Detailed technology deep dive, see our [blog post](https://www.deepspeed.ai/2020/05/27/fastest-bert-training.html).
* Tutorial on how to reproduce our results, see our [BERT pre-training tutorial](https://www.deepspeed.ai/tutorials/bert-pretraining/).
* The source code for our transformer kernels can be found in the [DeepSpeed repo](https://github.com/microsoft/deepspeed) and BERT pre-training code can be found in the [DeepSpeedExamples repo](https://github.com/microsoft/deepspeedexamples).
